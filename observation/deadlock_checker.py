from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import TrainState
from flatland.envs.fast_methods import fast_count_nonzero

from utils.render import render_env

class DeadlockChecker:
    def __init__(self, env):
        """Deadlock checker.

        Args:
            env (RailEnvWrapper): the environment to check for deadlocks
        """
        self.env = env
        self.agent_deadlock = [False] * self.env.number_of_agents
        # self.has_rendered = False   # [DEBUG] debugging purposes, for rendering deadlocks
        
    def reset(self):
        """Reset the deadlock checker.
        """
        self.agent_deadlock = [False] * self.env.number_of_agents
        # self.has_rendered = False   # [DEBUG] debugging purposes, for rendering deadlocks

    def update_deadlocks(self):
        """Update the deadlocks in the environment.
        """
        # build a map of active agents
        active_agents_positions = {}
        for agent in self.env.agents:
            if TrainState.MOVING <= agent.state <= TrainState.MALFUNCTION:  # i.e. the agent is on the map
                active_agents_positions[agent.position] = agent.handle
        
        self.can_move = [False] * len(self.env.agents)
        self.followers_graph = {handle: [] for handle in range(len(self.env.agents))}   # for each agent, store the agents that depend on it, i.e. the ones that are following it to see if they have the chance of moving where he is now
        for agent in self.env.agents:
            # check each "connected component" of active agents one by one
            if TrainState.MOVING <= agent.state <= TrainState.MALFUNCTION and not self.agent_deadlock[agent.handle]:
                self._check_and_build_graph(agent, active_agents_positions)

        # propagate can_move to followers
        self._propagate_can_move()

        # update deadlocks
        for agent in self.env.agents:
            if TrainState.MOVING <= agent.state <= TrainState.MALFUNCTION and not self.can_move[agent.handle]:
                self.agent_deadlock[agent.handle] = True

        # [DEBUG] render deadlocks
        # num_deadlocks = sum(self.agent_deadlock)
        # if num_deadlocks % 2 == 1 and not self.has_rendered:
        #     for agent in self.env.agents:
        #         if self.agent_deadlock[agent.handle]:    
        #             self.env.dev_obs_dict[agent.handle] = [agent.position]
        #         else:
        #             self.env.dev_obs_dict[agent.handle] = []
        #     render_env(self.env)
        #     self.has_rendered = True
            

    def _check_and_build_graph(self, agent, active_agents_positions):
        possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
        num_transitions = fast_count_nonzero(possible_transitions)
        if num_transitions == 0:
            # agent is in a dead-end, NOT a deadlock since it can invert its direction
            self.can_move[agent.handle] = True
            return

        # check if the agent can move by checking over all possible directions
        is_following_someone = False
        for dir in range(4):
            if possible_transitions[dir] == 0:
                continue

            new_pos = get_new_position(agent.position, dir)
            opposite_agent_handle = active_agents_positions[new_pos] if new_pos in active_agents_positions.keys() else -1
            if opposite_agent_handle == -1:
                # no agent in the new position, the agent can move
                self.can_move[agent.handle] = True
                return

            if self.agent_deadlock[opposite_agent_handle]:
                continue    # the opposite agent is in a deadlock, this direction is blocked, check other directions

            # otherwise we don't know, so we mark this agent as a follower of the opposite agent
            self.followers_graph[opposite_agent_handle].append(agent.handle)
            is_following_someone = True

        if not is_following_someone:
            # agent has no dependencies (i.e. is not following any other train's possible ability to move) nor can move anywhere, it is in a deadlock
            self.agent_deadlock[agent.handle] = True

    def _propagate_can_move(self):
        # propagate can_move to followers, by running a dfs on each "connected component" of agents as defined by the followers_graph
        visited = [False] * len(self.env.agents)
        for handle in range(len(self.env.agents)):
            if not visited[handle] and self.can_move[handle]:
                self._dfs_propagate_can_move(handle, visited)

    def _dfs_propagate_can_move(self, handle, visited):
        # dfs to propagate can_move to followers
        stack = [handle]
        while len(stack) > 0:
            agent_handle = stack.pop()
            if visited[agent_handle]:
                continue
            visited[handle] = True
            for follower in self.followers_graph[agent_handle]:
                if not visited[follower] and not self.can_move[follower]:
                    self.can_move[follower] = True
                    stack.append(follower)