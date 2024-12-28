from typing import Dict

from flatland.envs.agent_utils import TrainState
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import EnvAgent

from utils.decision_cells import find_decision_cells
from observation.deadlock_checker import DeadlockChecker


class RailEnvWrapper:
    def __init__(self, env: RailEnv):
        """Class constructor

        Args:
            env (RailEnv): the flatland environment to wrap
        """
        self.env = env

        self._decision_cells = None
        self.deadlock_checker = DeadlockChecker(env)

    def reset(self, **kwargs):
        """Reset the environment. 
        Builds the set of decision cells and resets the deadlock checker. Cycles until at least one agent is ready to depart.

        Returns:
            obs, info: the first observation and info dict from the environment where at least one agent is ready to depart
        """
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

        Furthermore, if all agents are either done or in deadlock, the environment is set to terminate in the next step.
        This early episode termination avoids useless computation and useless steps in the training batches.

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

        # check if all agents are either done or in deadlock
        all_done_or_deadlock = True
        for agent in self.env.agents:
            if agent.state != TrainState.DONE or not self.deadlock_checker.agent_deadlock[agent.handle]:
                all_done_or_deadlock = False
                break
        # if so, set the environment to terminate in the next step
        if all_done_or_deadlock:
            self.env._elapsed_steps = self.env._max_episode_steps
        
        return obs, reward, done, info

    def _on_decision_cell(self, agent: EnvAgent):
        return agent.position in self._decision_cells[agent.direction]

    def __getattr__(self, name):
        return getattr(self.env, name)

    def custom_reward(self, done, old_info, info):   
        """Compute the custom reward function.
        The custom reward function is the sum of the following components:
            1) a penalty for the agents that did not arrive at their target in time, if the episode is terminated.
                This is computed as the sum of the penalties for each agent, as computed by the standard environment (env._handle_end_reward),
                divided by the number of agents and the maximum number of steps in the episode (to normalize)
            2) 0.1 * (the number of agents that have just departed / the total number of agents)
            3) 5 * (the number of agents that have just arrived WITH A PENALTY FOR ARRIVING LATE / the total number of agents).
                The penalty for arriving late is computed as the number of steps that the agent arrived late, normalized by the number of steps in the episode
            4) -2.5 * (the number of agents that have just entered deadlock / the total number of agents)

        Args:
            done (dict): the dictionary returned by the env containing the done flag for each agent + the __all__ flag
            old_info (dict): the info dict from the previous step
            info (dict): the info dict from the current step

        Returns:
            float: the custom reward function value
        """

        no_arrival_penalty = 0
        newly_departed_agents = 0
        newly_arrived_agents = 0
        for handle in info["state"].keys():
            agent = self.env.agents[handle]
            if old_info["state"][handle] < TrainState.MOVING and info["state"][handle] >= TrainState.MOVING:
                newly_departed_agents += 1
            if old_info["state"][handle] < TrainState.DONE and info["state"][handle] == TrainState.DONE:
                newly_arrived_agents += 1 - max(agent.arrival_time - agent.latest_arrival, 0) / self.env._max_episode_steps
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
       
    def normalized_reward(self, reward):
        """Computes the true episodic reward as defined by Flatland, i.e. the normalized sum of the rewards of all agents.

        Args:
            reward (dict): the reward dictionary returned by the environment

        Returns:
            float: the true episodic reward
        """
        return 1.0 + sum(reward.values()) / self.env.get_num_agents() / self.env._max_episode_steps   

