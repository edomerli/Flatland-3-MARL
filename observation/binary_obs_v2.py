from typing import List, Optional, Any
import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.step_utils.states import TrainState
from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax
from flatland.envs.observations import TreeObsForRailEnv
from flatland.core.env_prediction_builder import PredictionBuilder

from utils.decision_cells import find_switches_and_switches_neighbors
from utils.render import render_env


class BinaryTreeObsV2(ObservationBuilder):
    def __init__(self):
        """Binary Tree Observation Builder.

        Args:
            max_depth (int): the maximum depth to explore the tree
        """
        self.max_depth = 2  # NOTE: we only explore up to depth 2, hard-coded
        self.agent_attr_dim = 14
        self.branch_dim = 10
        self.observation_dim = self.agent_attr_dim + 16 * self.branch_dim
        self.deadlock_checker = None
        
        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.location_has_target = None

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
        self.all_switches = set()
        for dir in self.switches.keys():
            self.all_switches.update(self.switches[dir])

        self.location_has_target = {tuple(agent.target): 1 for agent in self.env.agents}
    
    def get_many(self, handles: Optional[List[int]] = []):
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.
        """
        self.location_has_agent = {}

        for _agent in self.env.agents:
            # if _agent.state == TrainState.READY_TO_DEPART or TrainState.MOVING <= _agent.state <= TrainState.MALFUNCTION: # reintroduce if you want to consider agents that are ready to depart as well as moving agents
            if TrainState.MOVING <= _agent.state <= TrainState.MALFUNCTION:
                self.location_has_agent[_agent.position] = _agent.handle

        observations = super().get_many(handles)

        # [DEBUG] uncomment for debugging purposes to see the observations rendered on the env
        # counter = 0
        # for agent in self.env.agents:
        #     if (TrainState.MOVING <= agent.state <= TrainState.MALFUNCTION):
        #         counter += 1
        # if counter >= len(self.env.agents)//2:
        #     render_env(self.env)

        return observations

    def get(self, handle: int = 0):
        agent_attr_obs = np.zeros(self.agent_attr_dim)
        agent = self.env.agents[handle]

        agent_done = False
        if TrainState.WAITING <= agent.state <= TrainState.MALFUNCTION_OFF_MAP:
            agent_virtual_position = agent.initial_position
        elif TrainState.MOVING <= agent.state <= TrainState.MALFUNCTION:
            agent_virtual_position = agent.position
        else:   # i.e. agent.state == TrainState.DONE
            agent_virtual_position = (-1, -1)
            agent_done = True

        # the rest of the observation is computed only if the agent is not done!
        if agent_done:
            observation = np.full(self.observation_dim, 0)
            observation[agent.state] = 1
            return  observation
        
        ### AGENT and CURRENT POSITION information
        # agent_attr_obs[0-6]: the agent's state (one-hot encoded)
        agent_attr_obs[agent.state] = 1
        # agent_attr_obs[7-10]: the agent's speed (one-hot encoded)
        agent_attr_obs[7 + agent.speed_counter.max_count] = 1
        # agent_attr_obs[11]: whether the agent is deadlocked
        agent_attr_obs[11] = self.deadlock_checker.agent_deadlock[handle]
        # agent_attr_obs[12]: whether the agent is on a switch
        agent_attr_obs[12] = int(agent_virtual_position in self.switches[agent.direction])
        # agent_attr_obs[13]: whether the agent is near a switch
        agent_attr_obs[13] = int(agent_virtual_position in self.switches_neighbors[agent.direction])

        # preliminary steps
        visited = set()
        visited.add(agent_virtual_position)

        distance_map = self.env.distance_map.get()

        per_dir_obs = self._bfs_explore(agent, agent_virtual_position, distance_map, visited)

        observation = np.concatenate([agent_attr_obs, per_dir_obs[0], per_dir_obs[1], per_dir_obs[2], per_dir_obs[3]])

        return observation
    
    def _bfs_explore(self, agent, position, distance_map, visited):
        current_cell_dist = distance_map[agent.handle, position[0], position[1], agent.direction]
        per_dir_obs = {dir: np.zeros(4*self.branch_dim) for dir in range(4)}

        possible_transitions = self.env.rail.get_transitions(*position, agent.direction)
        num_transitions = fast_count_nonzero(possible_transitions)
        orientation = agent.direction
        if num_transitions == 1:
            orientation = fast_argmax(possible_transitions)

        # initialize the queue with the first neighboring cells
        queue = []  # queue holds node made of (position, first_turn, second_turn, new_direction, depth)
        for turn_dir, new_direction in enumerate([(orientation + turn_dir) % 4 for turn_dir in range(-1, 3)]):
            new_position = get_new_position(position, new_direction)
            if possible_transitions[new_direction]:
                per_dir_obs[turn_dir][0] = 1
                new_pos_dist = distance_map[agent.handle, new_position[0], new_position[1], new_direction]
                if not np.isinf(new_pos_dist):
                    per_dir_obs[turn_dir][1] = 1
                    per_dir_obs[turn_dir][2] = int(new_pos_dist < current_cell_dist)
                queue.append((new_position, turn_dir, -999, new_direction, 1))

        while queue:
            position, first_turn, second_turn, direction, depth = queue.pop(0)
            if position in visited:
                continue
            visited.add(position)

            # compute the future direction when at the (potential) switch
            next_dir = direction
            possible_transitions = self.env.rail.get_transitions(*position, direction)
            num_transitions = fast_count_nonzero(possible_transitions)
            if num_transitions == 1:
                next_dir = fast_argmax(possible_transitions)

            # check if there is an agent at the current position
            if position in self.location_has_agent:
                other_handle = self.location_has_agent[position]
                other_agent = self.env.agents[other_handle]

                base_idx = (depth==2) * 10 * second_turn
                # check if opposite direction or not
                if self._reverse_dir(next_dir, other_agent.direction):
                    per_dir_obs[first_turn][base_idx + 3] = 1
                else:
                    per_dir_obs[first_turn][base_idx + 4] = 1
                    # if same dir, check if agent is faster/equally fast than other agent or not
                    if agent.speed_counter.speed >= other_agent.speed_counter.speed:
                        per_dir_obs[first_turn][base_idx + 5] = 1

                # check if other agent is in deadlock
                if self.deadlock_checker.agent_deadlock[other_handle]:
                    per_dir_obs[first_turn][base_idx + 6] = 1
                # check if other agent is ready to depart
                if other_agent.state == TrainState.READY_TO_DEPART:
                    per_dir_obs[first_turn][base_idx + 7] = 1
                # check if other agent is malfunctioning
                if other_agent.malfunction_handler.malfunction_down_counter > 0:
                    per_dir_obs[first_turn][base_idx + 8] = 1

                continue    # NOTE: stop exploring since we found an agent

            # check if the agent's target is at the current position
            if position == agent.target:
                idx = 9 + (depth==2) * 10 * second_turn
                per_dir_obs[first_turn][idx] = 1
                # continue exploring though!

            # explore next cell/cells
            if position in self.switches[direction]:
                # I reached a switch
                # start next depth explorations if haven't reached max depth
                if depth == self.max_depth:
                    continue    # don't explore further
                for turn_dir, new_direction in enumerate([(direction + turn_dir) % 4 for turn_dir in range(-1, 2)]):    # NOTE: only 3 directions because we don't want to look backwards here
                    new_position = get_new_position(position, new_direction)
                    base_idx = 10 + 10 * turn_dir
                    if possible_transitions[new_direction]:
                        per_dir_obs[first_turn][base_idx] = 1
                        new_pos_dist = distance_map[agent.handle, new_position[0], new_position[1], new_direction]
                        if not np.isinf(new_pos_dist):
                            per_dir_obs[first_turn][base_idx + 1] = 1
                            per_dir_obs[first_turn][base_idx + 2] = int(new_pos_dist < current_cell_dist)
                        queue.append((new_position, first_turn, turn_dir, new_direction, depth + 1))
            else:
                # only one possible transition, continue in the same direction, don't increase depth
                new_position = get_new_position(position, next_dir)
                queue.append((new_position, first_turn, second_turn, next_dir, depth))

        # [DEBUG] uncomment for debugging purposes to see the observations rendered on the env
        if TrainState.MOVING <= agent.state <= TrainState.MALFUNCTION:
            self.env.dev_obs_dict[agent.handle] = list(visited)
        else:
            self.env.dev_obs_dict[agent.handle] = []
        return per_dir_obs
    
    def _reverse_dir(self, dir1, dir2):
        """Returns whether dir2 is the reverse direction of dir1.

        Args:
            dir1 (int): the first direction
            dir2 (int): the second direction
        """
        return dir2 == ((dir1 + 2) % 4)

    def _is_rail_at_position(self, position):
        """Returns whether there is a rail at the given position.

        Args:
            position (int, int): the position to check

        Returns:
            bool: whether there is a rail at the given position
        """
        row, col = position
        if row < 0 or row >= self.env.height or col < 0 or col >= self.env.width:
            return False
        transition_type = self.env.rail.get_full_transitions(row, col)
        return transition_type != 0
