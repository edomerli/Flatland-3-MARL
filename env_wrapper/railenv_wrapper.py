from typing import Dict

from flatland.envs.agent_utils import TrainState
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import EnvAgent
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_count_nonzero

from utils.conversions import dict_to_tensor
from utils.decision_cells import find_decision_cells
from observation.deadlock_checker import DeadlockChecker


class RailEnvWrapper:
    def __init__(self, env: RailEnv):
        self.env = env

        self._decision_cells = None
        self.deadlock_checker = DeadlockChecker(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._decision_cells = find_decision_cells(self.env)
        self.deadlock_checker.reset()

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
            if info["action_required"][agent.handle] and (TrainState.MOVING <= agent.state <= TrainState.STOPPED):  # i.e. condition 2) above
                info["action_required"][agent.handle] = self._on_decision_cell(agent)
        # TODO: action_required is false if the agent is in a deadlock?? Try it!    
        
        return obs, reward, done, info

    def _on_decision_cell(self, agent: EnvAgent):
        return agent.position in self._decision_cells[agent.direction]

    def __getattr__(self, name):
        return getattr(self.env, name)

    def custom_reward(self, done, old_info, info):        
        no_arrival_penalty = 0
        newly_departed_agents = 0
        newly_arrived_agents = 0
        for handle in info["state"].keys():
            agent = self.env.agents[handle]
            if old_info["state"][handle] < TrainState.MOVING and info["state"][handle] >= TrainState.MOVING:
                newly_departed_agents += 1
            if old_info["state"][handle] < TrainState.DONE and info["state"][handle] == TrainState.DONE:
                newly_arrived_agents += 1 - max(agent.arrival_time - agent.latest_arrival, 0) / (self.env._max_episode_steps - agent.latest_arrival + 1)    # +1 to avoid division by zero
                # TODO: remove
                if agent.arrival_time - agent.latest_arrival > 0:
                    print(f"Agent {handle} arrived {agent.arrival_time - agent.latest_arrival} steps late. Newly arrived agents: {newly_arrived_agents}")
            if done["__all__"] and agent.state != TrainState.DONE:
                no_arrival_penalty += self.env._handle_end_reward(agent)

        # count number of True values in self.deadlock_checker.agent_deadlock
        old_deadlocks_count = sum(self.deadlock_checker.agent_deadlock)
        self.deadlock_checker.update_deadlocks()
        deadlocks_count = sum(self.deadlock_checker.agent_deadlock)
        new_deadlocks = deadlocks_count - old_deadlocks_count

        N = self.env.get_num_agents()

        # no_arrival_penalty is normalized in the same way that the true episodic reward is
        return no_arrival_penalty / N / self.env._max_episode_steps + 0.1 * (newly_departed_agents / N) + 5 * (newly_arrived_agents / N) - 2.5 * (new_deadlocks / N)
       
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

        # TODO: remove
        all_done_or_deadlock = True
        for agent in self.env.agents:
            if agent.state == TrainState.DONE or self.deadlock_checker.agent_deadlock[agent.handle]:
                continue
            else:
                all_done_or_deadlock = False
                break

        n_deadlocks = sum(self.deadlock_checker.agent_deadlock)
        if all_done_or_deadlock and n_deadlocks > 0:
            print(f"Number of deadlock agents: {n_deadlocks}")
            print(f"Number of done agents: {sum([agent.state == TrainState.DONE for agent in self.env.agents])}")
            print(f"Steps done: {self.env._elapsed_steps}/{self.env._max_episode_steps}. Left: {self.env._max_episode_steps - self.env._elapsed_steps}")
            print(f"Percentage of steps left: {(self.env._max_episode_steps - self.env._elapsed_steps) / self.env._max_episode_steps}")


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

