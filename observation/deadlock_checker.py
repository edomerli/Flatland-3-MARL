import numpy as np
from enum import IntEnum

from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import TrainState
from flatland.envs.fast_methods import fast_count_nonzero


class DeadlockChecker:
    def __init__(self, env):
        self.env = env
        self.agent_deadlock = [False] * self.env.number_of_agents
        
    def reset(self):
        self.agent_deadlock = [False] * self.env.number_of_agents

    def update_deadlocks(self):
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
                self._check_and_build_graph(agent)

        # propagate can_move to followers
        self._propagate_can_move()

        # update deadlocks
        for agent in self.env.agents:
            if TrainState.MOVING <= agent.state <= TrainState.MALFUNCTION and not self.can_move[agent.handle]:
                self.agent_deadlock[agent.handle] = True

    def _check_and_build_graph(self, agent):
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
            opposite_agent_handle = self.env.agent_positions[new_pos] if new_pos in self.env.agent_positions else -1
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
        visited = [False] * len(self.env.agents)
        for handle in range(len(self.env.agents)):
            if not visited[handle] and self.can_move[handle]:
                self._dfs_propagate_can_move(handle, visited)

    def _dfs_propagate_can_move(self, handle, visited):
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
                
        

    # TODO: remove
    # def _check_deadlock(self, agent):
    #     self.checked_status[agent.handle] = CheckStatus.BEING_CHECKED   # being checked
        
    #     possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
    #     num_transitions = fast_count_nonzero(possible_transitions)
    #     if num_transitions == 0:
    #         # agent is in a dead-end, NOT a deadlock since it can invert its direction
    #         self.checked_status[agent.handle] = CheckStatus.CHECKED
    #         return 
        
    #     # check if the agent is in a deadlock by checking over all possible transitions
    #     for dir in range(4):
    #         if possible_transitions[dir] == 0:
    #             continue    # no transition in this direction, check other directions

    #         new_pos = get_new_position(agent.position, dir)
    #         opposite_agent_handle = self.env.agent_positions[new_pos] if new_pos in self.env.agent_positions else -1
    #         if opposite_agent_handle == -1:
    #             self.checked_status[agent.handle] = CheckStatus.CHECKED   # checked finished, no deadlock since the agent can move in this direction
    #             return 

    #         if self.checked_status[opposite_agent_handle] == CheckStatus.NOT_CHECKED:
    #             self._check_deadlock(self.env.agents[opposite_agent_handle])

    #         if self.checked_status[opposite_agent_handle] == CheckStatus.CHECKED:
    #             if self.agent_deadlock[opposite_agent_handle]:
    #                 continue    # the opposite agent is in a deadlock, this direction is blocked, check other directions
    #             else:
    #                 # the opposite agent is not in a deadlock, no deadlock since this direction might become available when the opposite agent moves
    #                 self.checked_status[agent.handle] = CheckStatus.CHECKED
    #                 return
            
    #         # the opposite agent is being checked, add a dependency in the graph
    #         self.followers_graph[agent.handle].append(opposite_agent_handle)

    #         # loop continues by checking the next direction

    #     if len(self.followers_graph[agent.handle]) == 0:
    #         # agent has no dependencies nor can move anywhere, it is in a deadlock
    #         self.checked_status[agent.handle] = CheckStatus.CHECKED
    #         self.agent_deadlock[agent.handle] = True

    # TODO: remove         
    # def _resolve_dependencies(self):
        # visit each agent using dfs
        # visited = [False] * len(self.env.agents)
        # stack = [agent.handle for agent in self.env.agents if self.checked_status[agent.handle] == CheckStatus.BEING_CHECKED]
        # while len(stack) > 0:
        #     agent_handle = stack.pop()
        #     if visited[agent_handle]:
        #         continue
        #     visited[agent_handle] = True
        #     for dep in self.followers_graph[agent_handle]:
        #         if self.checked_status[dep] == CheckStatus.CHECKED and not self.agent_deadlock[dep]:
        #             self.checked_status[agent_handle] = CheckStatus.CHECKED
        #         elif self.checked_status[dep] == CheckStatus.BEING_CHECKED and not visited[dep]:
        #             stack.append(dep)