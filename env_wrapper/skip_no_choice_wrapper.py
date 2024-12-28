from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import TrainState
import copy


class SkipNoChoiceWrapper:
    def __init__(self, env: RailEnv):
        self.env = env
    
    def step(self, action_dict):
        while True:
            old_info = self.env.get_info_dict()
            obs, reward, done, info = self.env.step(action_dict)

            # exit if the episode has terminated
            if done["__all__"]:
                return obs, reward, done, info
            
            # conditions for exiting because of the non-null custom reward function
            for handle in info["state"].keys():
                # 1. an agent has departed
                if old_info["state"][handle] < TrainState.MOVING and info["state"][handle] >= TrainState.MOVING:
                    return obs, reward, done, info
                # 2. an agent has reached its target
                if old_info["state"][handle] < TrainState.DONE and info["state"][handle] == TrainState.DONE:
                    return obs, reward, done, info
                
            # 3. an agent has reached deadlock
            old_deadlock_checker = copy.deepcopy(self.env.deadlock_checker)
            old_deadlocks_count = sum(old_deadlock_checker.agent_deadlock)
            self.env.deadlock_checker.update_deadlocks()
            deadlocks_count = sum(self.env.deadlock_checker.agent_deadlock)
            new_deadlocks = deadlocks_count - old_deadlocks_count

            # IMPORTANT: reset the deadlock checker to the old one!
            self.env.deadlock_checker = old_deadlock_checker

            if new_deadlocks > 0:
                return obs, reward, done, info
                
            # condition for exiting because at least one agent has a choice to make
            for agent in self.env.agents:
                if info["action_required"][agent.handle]:
                    return obs, reward, done, info
            
            # otherwise: make agents take the default, DO_NOTHING, move at next step
            action_dict = {}

    def __getattr__(self, name):
        return getattr(self.env, name)