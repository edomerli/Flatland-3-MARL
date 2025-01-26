from typing import Dict

from flatland.envs.agent_utils import TrainState
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import EnvAgent

from utils.decision_cells import find_decision_cells
from observation.deadlock_checker import DeadlockChecker


class TestRailEnvWrapper:
    def __init__(self, env: RailEnv):
        """Class constructor

        Args:
            env (RailEnv): the flatland environment to wrap
        """
        self.env = env

        self._decision_cells = None
        self.deadlock_checker = DeadlockChecker(env)

        self.some_agent_departed = False

    def reset_components(self, **kwargs):
        """Reset the deadlock checker and the observation builder components of the environment.
        """
        self._decision_cells = find_decision_cells(self.env)
        # associate the deadlock checker to the observation builder
        self.env.obs_builder.set_deadlock_checker(self.deadlock_checker)
        self.deadlock_checker.reset()
        self.env.obs_builder.reset()
        
    def step(self, action_dict: Dict[int, RailEnvActions]):
        """Step the environment with the given actions.

        Overrides the action_required info in the info dict s.t.
        Before:
            info["action_required"][agent_id] is True iff
                1) the agent is in a READY_TO_DEPART state
                OR
                2) the agent is on the map (i.e. either MOVING, STOPPED or MALFUNCTION) AND is at a cell entry point

        After:
            info["action_required"][agent_id] is True iff
                1) ...same as above...
                OR 
                2) ...same as above... AND the agent is on a decision cell (i.e. a cell where the agent can choose its direction!) AND is not in deadlock

        Args:
            action_dict (Dict[int, RailEnvActions]): the actions to take for each agent

        Returns:
            obs, reward, done, info: the next observation, reward, done flag and info dict from the environment. Note: only the info dict is modified
        """
        obs, reward, done, info = self.env.step(action_dict)

        # update action_required in the info dict
        for agent in self.env.agents:
            if info["action_required"][agent.handle] and (TrainState.MOVING <= agent.state <= TrainState.STOPPED):  # i.e. condition 2) above
                info["action_required"][agent.handle] = self._on_decision_cell(agent) and not self.deadlock_checker.agent_deadlock[agent.handle]
        
        return obs, reward, done, info
    
    def all_agents_waiting(self):
        for agent in self.env.agents:
            if agent.state != TrainState.WAITING:
                return False
        return True 
    
    def all_done_or_deadlock(self):
        for agent in self.env.agents:
            if agent.state != TrainState.DONE or not self.deadlock_checker.agent_deadlock[agent.handle]:
                return False
        return True
        
    def _on_decision_cell(self, agent: EnvAgent):
        return agent.position in self._decision_cells[agent.direction]

    def __getattr__(self, name):
        return getattr(self.env, name)
