from typing import Dict

from flatland.envs.agent_utils import TrainState
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_env import RailEnv

from utils.conversions import dict_to_tensor

class RailEnvWrapper:
    def __init__(self, env: RailEnv):
        self.env = env

        # TODO: remove?
        # self.departed_agents = set()
        # self.arrived_agents = set()
        self.deadlock_agents = set()

    # TODO: remove? lo potrei fare se volessi tenermi gli agenti (partiti e) done, a fini di passarli a NN o custom_reward
    # def step(self, action_dict_: Dict[int, RailEnvActions]):
    #     obs, rews, dones, infos = self.env.step(action_dict_)

    #     self.n_departed_agents_old = len(self.departed_agents)
    #     self.n_arrived_agents_old = len(self.arrived_agents)

    #     self.departed_agents = self.departed_agents.union({i for i, info in infos.items() if info["state"] >= TrainState.MOVING})
    #     self.arrived_agents = self.arrived_agents.union({i for i, info in infos.items() if info["state"] == TrainState.DONE})

    #     return obs, rews, dones, infos
        

    # TODO: attivalo solo se ne hai bisogno, ma faccio già che prima di ogni step, se non c'è nessun agente pronto a partire, cicla fino a quando non ce n'è almeno uno
    # def reset(self, cycle_until_action_required=True):
    #     print("Usato custom reset")
    #     obs_dict, info_dict = self.env.reset()

    #     if cycle_until_action_required:
    #         obs_dict, info_dict = self.cycle_until_action_required(info_dict)

    #     return obs_dict, info_dict

    # TODO: remove        
    # def reset(self, **kwargs):
    #     obs, info = self.env.reset(**kwargs)
    #     self.env.obs_builder.dead_lock_avoidance_agent.reset(self.env)
    #     return obs, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # cycle until at least one agent is ready to depart
        while not any(info["action_required"].values()):
            obs, _, _, info = self.env.step({})  # empty dict === RailEnv will choose DO_NOTHING action by default
        return obs, info

    def __getattr__(self, name):
        return getattr(self.env, name)

    # TODO: remove
    # def cycle_until_action_required(self, obs, info_dict):
    #     obs_dict = None
    #     while not any(info_dict["action_required"].values()):
    #         obs_dict, _, _, info_dict = self.env.step({})   # empty dict === RailEnv will choose DO_NOTHING action by

    #     if obs_dict is not None:
    #         obs = dict_to_tensor(obs_dict)
    #     return obs, info_dict

    def custom_reward(self, done, reward, old_info, info):
        normalized_reward = self.normalized_reward(done, reward)
        
        newly_departed_agents = 0
        newly_arrived_agents = 0
        for handle in info["state"].keys():
            if old_info["state"][handle] < TrainState.MOVING and info["state"][handle] >= TrainState.MOVING:
                newly_departed_agents += 1
            if old_info["state"][handle] < TrainState.DONE and info["state"][handle] == TrainState.DONE:
                newly_arrived_agents += 1

        # count NEW(!) deadlocks
        deadlock_avoidance_agent = self.env.obs_builder.dead_lock_avoidance_agent
        agent_can_move = deadlock_avoidance_agent.agent_can_move

        new_deadlocks = 0
        for handle, state in info["state"].items():
            if handle in self.deadlock_agents:
                continue
            elif (TrainState.MOVING <= state <= TrainState.MALFUNCTION) and (agent_can_move.get(handle, None) is None):
                new_deadlocks += 1
                self.deadlock_agents.add(handle)

        N = self.env.get_num_agents()

        return normalized_reward + 0.1 * (newly_departed_agents / N) + 5 * (newly_arrived_agents / N) - 2.5 * (new_deadlocks / N)

        
    def normalized_reward(self, done, reward):
        if self.is_done(done, self.env.get_info_dict()):
            if not done["__all__"]:
                # if not all agents are done, but the assert above passed, it means that some agents are done and some are in deadlock
                # -> we make the episode finish earlier by setting elapsed steps to max_episode and use it to compute the end of episode reward.
                # This way we don't pollute the replay buffer with useless transitions
                self.env._elapsed_steps = self.env._max_episode_steps
                for agent in self.env.agents:
                    reward[agent.handle] = self.env._handle_end_reward(agent)

        # normalized_reward = 1 + sum_of_agents_reward / (num_agents * T_max)
        return 1.0 + sum(reward.values()) / self.env.get_num_agents() / self.env._max_episode_steps   

    def is_done(self, done, info):
        if done["__all__"]:
            return True
        
        deadlock_avoidance_agent = self.env.obs_builder.dead_lock_avoidance_agent
        agent_can_move = deadlock_avoidance_agent.agent_can_move
        # print(agent_can_move)
        
        all_done_or_deadlock = True
        for handle, state in info["state"].items():
            if state == TrainState.DONE:
                # print(f"Agent {handle} is done")
                continue
            elif (TrainState.MOVING <= state <= TrainState.MALFUNCTION) and (agent_can_move.get(handle, None) is None):
                # print(f"Agent {handle} is in deadlock, state: {state}")
                # agent is in deadlock
                continue
            else:
                # print(f"Agent {handle} is not done and not in deadlock")
                all_done_or_deadlock = False
                break
        
        return all_done_or_deadlock

