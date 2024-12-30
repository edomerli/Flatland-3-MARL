from typing import List, Optional, Any
import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.step_utils.states import TrainState
from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax

from utils.decision_cells import find_switches_and_switches_neighbors
from utils.render import render_env


class BinaryTreeObs(ObservationBuilder):
    def __init__(self, max_depth: Any):
        """Binary Tree Observation Builder.

        Args:
            max_depth (int): the maximum depth to explore the tree
        """
        self.max_depth = max_depth
        self.observation_dim = 40
        self.deadlock_checker = None

    def set_deadlock_checker(self, deadlock_checker):
        """Set the deadlock checker.

        Args:
            deadlock_checker (deadlock_checker.DeadlockChecker): the deadlock checker to use
        """
        self.deadlock_checker = deadlock_checker

    def reset(self):
        """Reset the observation builder.
        """
        self.switches, self.switches_neighbors = find_switches_and_switches_neighbors(self.env)

    def get_many(self, handles: Optional[List[int]] = None):
        """Get the observations for a list of agents.

        Args:
            handles (Optional[List[int]], optional): the list of agents's handles for which to compute the observations. Defaults to None.

        Returns:
            observations: the list of observations for the agents
        """
        self.agents_target = self._get_moving_agents_targets()
        observations = super().get_many(handles)
        return observations

    def get(self, handle: int = 0):
        """Get the observation for an agent.

        An observation is defined as following (all values are BINARY):
        - observation[0-6]: the agent's state (one-hot encoded)
        - observation[7]: current agent is located at a switch, where it can take a routing decision
        - observation[8]: current agent is located at a switch, could be either a switch where it can take a decision or not (i.e. only one path)
        - observation[9]: current agent is located one step *before* a switch, at which switch it can take a routing decision
        - observation[10]: current agent is located one step *before* a switch, could be either a switch where it can take a decision or not (i.e. only one path)
        - observation[11-14]: for each direction, whether there is a path towards the target (one-hot encoded)
        - observation[15-18]: for each direction, whether the path takes us closer to the target (one-hot encoded) (0 if the path is longer or there is no path)
        - observation[19-22]: for each direction, whether there is a path and there is an agent with opposite direction OR an agent with same direction but in a deadlock on this path (one-hot encoded)
        - observation[23-26]: for each direction, whether there is a path and there is an agent with same direction AND not in a deadlock on this path (one-hot encoded)
        - observation[27-30]: for each direction, whether there is a path and the agent's target is on this path (one-hot encoded)
        - observation[31-34]: for each direction, whether there is a path and there is another agent's target on this path (one-hot encoded)
        - observation[35-38]: the agent's speed (one-hot encoded)
        - observation[39]: whether the agent is deadlocked
        
        Args:
            handle (int, optional): the handle of the agent for which to compute the observation. Defaults to 0.

        Returns:
            observation: the observation for the agent
        """
        observation = np.zeros(self.observation_dim)
        agent = self.env.agents[handle]

        # compute the agent's virtual position
        agent_done = False
        if TrainState.WAITING <= agent.state <= TrainState.MALFUNCTION_OFF_MAP:
            agent_virtual_position = agent.initial_position
        elif TrainState.MOVING <= agent.state <= TrainState.MALFUNCTION:
            agent_virtual_position = agent.position
        else:   # i.e. agent.state == TrainState.DONE
            agent_virtual_position = (-1, -1)
            agent_done = True
        # observation[0-6]: the agent's state (one-hot encoded)
        observation[agent.state] = 1

        # the rest of the observation is computed only if the agent is not done!
        if not agent_done:
            # compute current distance and adjust orientation if the next direction is already defined because it's the only possible transition
            distance_map = self.env.distance_map.get()
            current_cell_dist = distance_map[handle,
                                             agent_virtual_position[0], agent_virtual_position[1],
                                             agent.direction]
            possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)
            orientation = agent.direction
            if fast_count_nonzero(possible_transitions) == 1:
                orientation = fast_argmax(possible_transitions)

            # compute the observation's features for each direction (-1: left, 0: forward, 1: right, 2: backward - at dead-ends)
            for dir_loop, branch_direction in enumerate([(orientation + dir_loop) % 4 for dir_loop in range(-1, 3)]):
                # consider only the directions that can be taken
                if possible_transitions[branch_direction]:
                    new_position = get_new_position(agent_virtual_position, branch_direction)
                    new_cell_dist = distance_map[handle,
                                                 new_position[0], new_position[1],
                                                 branch_direction]
                    # observation[15-18]: for each direction, whether the path takes us closer to the target (one-hot encoded) (0 if the path is longer or there is no path)
                    if not (np.math.isinf(new_cell_dist) and np.math.isinf(current_cell_dist)):
                        observation[16 + dir_loop] = int(new_cell_dist < current_cell_dist)

                    has_opp_agent, has_same_agent, has_target, has_opp_target, min_dist = self._explore(handle,
                                                                                                           new_position,
                                                                                                           branch_direction,
                                                                                                           distance_map)

                    # observation[11-14]: for each direction, whether there is a path towards the target (one-hot encoded)
                    if not (np.math.isinf(min_dist) and np.math.isinf(current_cell_dist)):
                        observation[12 + dir_loop] = int(min_dist < current_cell_dist)
                    # observation[19-22]: for each direction, whether there is a path and there is an agent with opposite direction on this path (one-hot encoded)
                    # observation[23-26]: for each direction, whether there is a path and there is an agent with same direction on this path (one-hot encoded)
                    # observation[27-30]: for each direction, whether there is a path and the agent's target is on this path (one-hot encoded)
                    # observation[31-34]: for each direction, whether there is a path and there is another agent's target on this path (one-hot encoded)
                    observation[20 + dir_loop] = has_opp_agent
                    observation[24 + dir_loop] = has_same_agent
                    observation[28 + dir_loop] = has_target
                    observation[32 + dir_loop] = has_opp_target

            agents_on_switch, \
            agents_near_to_switch, \
            agents_near_to_switch_all, \
            agents_on_switch_all = \
                self._on_or_near_switch(agent_virtual_position, agent.direction)

            # observation[7]: current agent is located at a switch, where it can take a routing decision
            # observation[8]: current agent is located at a switch, could be either a switch where it can take a decision or not (i.e. only one path)
            # observation[9]: current agent is located one step *before* a switch, at which switch it can take a routing decision
            # observation[10]: current agent is located one step *before* a switch, could be either a switch where it can take a decision or not (i.e. only one path)
            observation[7] = int(agents_on_switch)
            observation[8] = int(agents_on_switch_all)
            observation[9] = int(agents_near_to_switch)
            observation[10] = int(agents_near_to_switch_all)

            # observation[35-38]: the agent's speed (one-hot encoded)
            observation[35 + agent.speed_counter.max_count] = 1     # max_count = int(1.0 / speed) - 1, i.e. 0 if speed==1, 1 if speed==0.5, 2 if speed==0.33, 3 if speed==0.25
            
            # observation[39]: whether the agent is deadlocked
            observation[39] = self.deadlock_checker.agent_deadlock[handle]

        observation[np.isinf(observation)] = -1
        observation[np.isnan(observation)] = -1

        return observation
    
    def _explore(self, handle, new_position, new_direction, distance_map, depth=0):
        has_opp_agent = 0
        has_same_agent = 0
        has_target = 0
        has_opp_target = 0
        min_dist = distance_map[handle, new_position[0], new_position[1], new_direction]

        # stop exploring (max_depth reached)
        if depth >= self.max_depth:
            return has_opp_agent, has_same_agent, has_target, has_opp_target, min_dist

        # we at most explore 100 cells
        cnt = 0
        while cnt < 100:
            cnt += 1

            opp_agent = self.env.agent_positions[new_position]
            if opp_agent != -1 and opp_agent != handle:
                if self.env.agents[opp_agent].direction != new_direction:
                    # agent with opposite direction found, stop exploring
                    has_opp_agent = 1
                    return has_opp_agent, has_same_agent, has_target, has_opp_target, min_dist
                else:
                    # agent with the same direction found, stop exploring
                    # NOTE: we consider an agent with the same direction but deadlocked as an agent with opposite direction!
                    if self.deadlock_checker.agent_deadlock[opp_agent]:
                        has_opp_agent = 1
                    else:
                        has_same_agent = 1
                    return has_opp_agent, has_same_agent, has_target, has_opp_target, min_dist

            # the switch refers to a switch where the agent *can* take a decision! Not any switch (i.e. maybe only for agents coming from other directions)
            agents_on_switch, _, _, _ = self._on_or_near_switch(new_position, new_direction)

            if new_position == self.env.agents[handle].target:
                has_target = 1
                return has_opp_agent, has_same_agent, has_target, has_opp_target, min_dist
            elif new_position in self.agents_target:
                has_opp_target = 1
                # continue exploring though!

            possible_transitions = self.env.rail.get_transitions(*new_position, new_direction)
            if agents_on_switch:
                orientation = new_direction
                possible_transitions_nonzero = fast_count_nonzero(possible_transitions)
                if possible_transitions_nonzero == 1:
                    orientation = fast_argmax(possible_transitions)

                for dir_loop, branch_direction in enumerate(
                        [(orientation + dir_loop) % 4 for dir_loop in range(-1, 3)]):
                    # branch the exploration path and aggregate the found information
                    if possible_transitions[dir_loop] == 1:
                        hoa, hsa, ht, hot, m_dist = self._explore(handle,
                                                                     get_new_position(new_position, dir_loop),
                                                                     dir_loop,
                                                                     distance_map,
                                                                     depth + 1)
                        has_opp_agent = max(hoa, has_opp_agent)
                        has_same_agent = max(hsa, has_same_agent)
                        has_target = max(has_target, ht)
                        has_opp_target = max(has_opp_target, hot)
                        min_dist = min(min_dist, m_dist)
                return has_opp_agent, has_same_agent, has_target, has_opp_target, min_dist
            else:
                new_direction = fast_argmax(possible_transitions)
                new_position = get_new_position(new_position, new_direction)

            min_dist = min(min_dist, distance_map[handle, new_position[0], new_position[1], new_direction])

        return has_opp_agent, has_same_agent, has_target, has_opp_target, min_dist
    
    def _on_or_near_switch(self, position, direction):
        agent_on_switch = position in self.switches[direction]
        agent_on_switch_all = False
        for dir in range(4):
            agent_on_switch_all |= position in self.switches[dir]
            if agent_on_switch_all:
                break

        agent_near_to_switch = False
        # look one step forward in each direction
        possible_transitions = self.env.rail.get_transitions(*position, direction)
        for new_dir in range(4):
            # check if you could move there
            if possible_transitions[new_dir]:
                # check if the new position is a switch
                new_pos = get_new_position(position, new_dir)
                if new_pos in self.switches[new_dir]:
                    agent_near_to_switch = True

        agent_near_to_switch_all = position in self.switches_neighbors[direction]
        
        return agent_on_switch, agent_near_to_switch, agent_near_to_switch_all, agent_on_switch_all

    def _get_moving_agents_targets(self):
        agent_targets = [agent.target for agent in self.env.agents if TrainState.MOVING <= agent.state <= TrainState.MALFUNCTION]
        return set(agent_targets)