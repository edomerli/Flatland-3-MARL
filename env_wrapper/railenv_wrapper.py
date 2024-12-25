from typing import Dict

from flatland.envs.agent_utils import TrainState
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import EnvAgent
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_count_nonzero

from utils.conversions import dict_to_tensor
from utils.decision_cells import find_decision_cells


class RailEnvWrapper:
    def __init__(self, env: RailEnv):
        self.env = env

        self._decision_cells = None
        self.deadlock_agents = set()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._decision_cells = find_decision_cells(self.env)   # TODO: do I even need this now???

        # cycle until at least one agent is ready to depart
        any_agent_ready_to_depart = False
        while True:
            for agent in self.env.agents:
                if agent.state == TrainState.READY_TO_DEPART:
                    any_agent_ready_to_depart = True
                    break
            if any_agent_ready_to_depart:
                break
            obs, _, _, info = self.env.step({})  # empty dict === RailEnv will choose DO_NOTHING action by default

        return obs, info
        
    def step(self, action_dict: Dict[int, RailEnvActions]):
        obs, reward, done, info = self.env.step(action_dict)

        # override action_required info
        # info["action_required"][agent_id] given from env.step() is True iff:
            # 1) the agent is in a READY_TO_DEPART state
            # OR
            # 2) the agent is on the map (i.e. either MOVING, STOPPED or MALFUNCTION) AND is at a cell entry point

        # we want to override this info s.t. info["action_required"][agent_id] is True iff:
            # 1) ...same as above...
            # OR 
            # 2) ...same as above... AND the agent is on a decision cell (i.e. a cell where the agent can choose its direction!)
        for agent in self.env.agents:
            agent_id = agent.handle

            if info["action_required"][agent_id] and (TrainState.MOVING <= agent.state <= TrainState.STOPPED):  # i.e. condition 2) above
                info["action_required"][agent_id] = self._on_decision_cell(agent)
            
        return obs, reward, done, info

    def _on_decision_cell(self, agent: EnvAgent):
        return agent.position in self._decision_cells[agent.direction]

    def __getattr__(self, name):
        return getattr(self.env, name)

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
        # TODO: restore but adjust
        # deadlock_avoidance_agent = self.env.obs_builder.dead_lock_avoidance_agent
        # agent_can_move = deadlock_avoidance_agent.agent_can_move

        # TODO: remove
        # _, properties, _ = self.env.obs_builder.get_properties()
        # deadlocked_agents = properties["deadlocked"]

        # TODO: restore
        # new_deadlocks = 0
        # for handle, state in info["state"].items():
        #     if handle in self.deadlock_agents:
        #         continue
        #     elif (TrainState.MOVING <= state <= TrainState.MALFUNCTION) and bool(deadlocked_agents[handle]): # TODO: remove -> (agent_can_move.get(handle, None) is None):
        #         new_deadlocks += 1
        #         self.deadlock_agents.add(handle)

        N = self.env.get_num_agents()

        # return normalized_reward + 0.1 * (newly_departed_agents / N) + 5 * (newly_arrived_agents / N) - 2.5 * (new_deadlocks / N)
        return normalized_reward + 0.1 * (newly_departed_agents / N) + 5 * (newly_arrived_agents / N)
        
    def normalized_reward(self, done, reward):
        # if self.is_done(done, self.env.get_info_dict()):
        #     if not done["__all__"]: # TODO: this will always be True if we call normalized_reward at the end of the episode! No matter what!! Instead should use agent.state == TrainState.DONE
        #         # if not all agents are done, but the assert above passed, it means that some agents are done and some are in deadlock
        #         # -> we make the episode finish earlier by setting elapsed steps to max_episode and use it to compute the end of episode reward.
        #         # This way we don't pollute the replay buffer with useless transitions
        #         self.env._elapsed_steps = self.env._max_episode_steps
        #         for agent in self.env.agents:
        #             reward[agent.handle] = self.env._handle_end_reward(agent)

        return 1.0 + sum(reward.values()) / self.env.get_num_agents() / self.env._max_episode_steps   

    def is_done(self, done, info):
        if done["__all__"]:
            return True
        return False    # TODO: remove if you want to use the deadlock checker
        
        _, properties, _ = self.env.obs_builder.get_properties()
        deadlocked_agents = properties["deadlocked"]
        
        # TODO: remove
        # deadlock_avoidance_agent = self.env.obs_builder.dead_lock_avoidance_agent
        # agent_can_move = deadlock_avoidance_agent.agent_can_move
        # print(agent_can_move)
        
        all_done_or_deadlock = True
        for handle, state in info["state"].items():
            if state == TrainState.DONE:
                # print(f"Agent {handle} is done")
                continue
            elif (TrainState.MOVING <= state <= TrainState.MALFUNCTION) and bool(deadlocked_agents[handle]): #TODO: remove -> (agent_can_move.get(handle, None) is None):
                # print(f"Agent {handle} is in deadlock, state: {state}, deadlocked_agents: {deadlocked_agents}")
                # exit()
                # agent is in deadlock
                continue
            else:
                # print(f"Agent {handle} is not done and not in deadlock")
                all_done_or_deadlock = False
                break
        
        return all_done_or_deadlock

