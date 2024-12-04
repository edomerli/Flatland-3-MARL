from typing import List, Optional, Any

import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.step_utils.states import TrainState
from flatland.envs.rail_env import RailEnvActions

from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax

from flatland_starter_kit.agent_can_choose_helper import AgentCanChooseHelper
from flatland_starter_kit.dead_lock_avoidance_agent import DeadLockAvoidanceAgent
from flatland_starter_kit.deadlock_check import get_agent_positions, get_agent_targets

"""
LICENCE for the FastTreeObs Observation Builder  

The observation can be used freely and reused for further submissions. Only the author needs to be referred to
/mentioned in any submissions - if the entire observation or parts, or the main idea is used.

Author: Adrian Egli (adrian.egli@gmail.com)

[Linkedin](https://www.researchgate.net/profile/Adrian_Egli2)
[Researchgate](https://www.linkedin.com/in/adrian-egli-733a9544/)
"""


class FastTreeObs(ObservationBuilder):

    def __init__(self, max_depth: Any):
        self.max_depth = max_depth
        self.observation_dim = 40
        self.agent_can_choose_helper = None
        self.dead_lock_avoidance_agent = DeadLockAvoidanceAgent(None, 4)  # 4 is the action space size, not considering the DO_NOTHING action

    # TODO: remove??
    def debug_render(self, env_renderer):
        agents_can_choose, agents_on_switch, agents_near_to_switch, agents_near_to_switch_all = \
            self.agent_can_choose_helper.required_agent_decision()
        self.env.dev_obs_dict = {}
        for a in range(max(3, self.env.get_num_agents())):
            self.env.dev_obs_dict.update({a: []})

        selected_agent = None
        if agents_can_choose[0]:
            if self.env.agents[0].position is not None:
                self.debug_render_list.append(self.env.agents[0].position)
            else:
                self.debug_render_list.append(self.env.agents[0].initial_position)

        if self.env.agents[0].position is not None:
            self.debug_render_path_list.append(self.env.agents[0].position)
        else:
            self.debug_render_path_list.append(self.env.agents[0].initial_position)

        env_renderer.gl.agent_colors[0] = env_renderer.gl.rgb_s2i("FF0000")
        env_renderer.gl.agent_colors[1] = env_renderer.gl.rgb_s2i("666600")
        env_renderer.gl.agent_colors[2] = env_renderer.gl.rgb_s2i("006666")
        env_renderer.gl.agent_colors[3] = env_renderer.gl.rgb_s2i("550000")

        self.env.dev_obs_dict[0] = self.debug_render_list
        self.env.dev_obs_dict[1] = self.agent_can_choose_helper.switches.keys()
        self.env.dev_obs_dict[2] = self.agent_can_choose_helper.switches_neighbours.keys()
        self.env.dev_obs_dict[3] = self.debug_render_path_list

    def reset(self):
        if self.agent_can_choose_helper is None:
            self.agent_can_choose_helper = AgentCanChooseHelper()
        self.agent_can_choose_helper.build_data(self.env)
        self.debug_render_list = []
        self.debug_render_path_list = []
        self.dead_lock_avoidance_agent.reset(self.env)

    def _explore(self, handle, new_position, new_direction, distance_map, depth=0):
        has_opp_agent = 0
        has_same_agent = 0
        has_target = 0
        has_opp_target = 0
        visited = []
        min_dist = distance_map[handle, new_position[0], new_position[1], new_direction]

        # stop exploring (max_depth reached)
        if depth >= self.max_depth:
            return has_opp_agent, has_same_agent, has_target, has_opp_target, visited, min_dist

        # max_explore_steps = 100 -> just to ensure that the exploration ends
        cnt = 0
        while cnt < 100:
            cnt += 1

            visited.append(new_position)
            opp_a = self.env.agent_positions[new_position]
            if opp_a != -1 and opp_a != handle:
                if self.env.agents[opp_a].direction != new_direction:
                    # opp agent found -> stop exploring. This would be a strong signal.
                    has_opp_agent = 1
                    return has_opp_agent, has_same_agent, has_target, has_opp_target, visited, min_dist
                else:
                    # same agent found
                    # the agent can follow the agent, because this agent is still moving ahead and there shouldn't
                    # be any dead-lock nor other issue -> agent is just walking -> if other agent has a deadlock
                    # this should be avoided by other agents -> one edge case would be when other agent has it's
                    # target on this branch -> thus the agents should scan further whether there will be an opposite
                    # agent walking on same track
                    has_same_agent = 1
                    # !NOT stop exploring!
                    return has_opp_agent, has_same_agent, has_target, has_opp_target, visited, min_dist

            # agents_on_switch == TRUE -> Current cell is a switch where the agent can decide (branch) in exploration
            # agent_near_to_switch == TRUE -> One cell before the switch, where the agent can decide
            #
            agents_on_switch, agents_near_to_switch, _, _ = \
                self.agent_can_choose_helper.check_agent_decision(new_position, new_direction)

            if agents_near_to_switch:
                # The exploration was walking on a path where the agent can not decide
                # Best option would be MOVE_FORWARD -> Skip exploring - just walking
                return has_opp_agent, has_same_agent, has_target, has_opp_target, visited, min_dist

            if self.env.agents[handle].target in self.agents_target:
                has_opp_target = 1

            if self.env.agents[handle].target == new_position:
                has_target = 1
                return has_opp_agent, has_same_agent, has_target, has_opp_target, visited, min_dist

            possible_transitions = self.env.rail.get_transitions(*new_position, new_direction)
            if agents_on_switch:
                orientation = new_direction
                possible_transitions_nonzero = fast_count_nonzero(possible_transitions)
                if possible_transitions_nonzero == 1:
                    orientation = fast_argmax(possible_transitions)

                for dir_loop, branch_direction in enumerate(
                        [(orientation + dir_loop) % 4 for dir_loop in range(-1, 3)]):
                    # branch the exploration path and aggregate the found information
                    # --- OPEN RESEARCH QUESTION ---> is this good or shall we use full detailed information as
                    # we did in the TreeObservation (FLATLAND) ?
                    if possible_transitions[dir_loop] == 1:
                        hoa, hsa, ht, hot, v, m_dist = self._explore(handle,
                                                                     get_new_position(new_position, dir_loop),
                                                                     dir_loop,
                                                                     distance_map,
                                                                     depth + 1)
                        visited.append(v)
                        has_opp_agent = max(hoa, has_opp_agent)
                        has_same_agent = max(hsa, has_same_agent)
                        has_target = max(has_target, ht)
                        has_opp_target = max(has_opp_target, hot)
                        min_dist = min(min_dist, m_dist)
                return has_opp_agent, has_same_agent, has_target, has_opp_target, visited, min_dist
            else:
                new_direction = fast_argmax(possible_transitions)
                new_position = get_new_position(new_position, new_direction)

            min_dist = min(min_dist, distance_map[handle, new_position[0], new_position[1], new_direction])

        return has_opp_agent, has_same_agent, has_target, has_opp_target, visited, min_dist

    def get_many(self, handles: Optional[List[int]] = None):
        self.dead_lock_avoidance_agent.start_step(False)
        self.agent_positions = get_agent_positions(self.env)
        self.agents_target = get_agent_targets(self.env)
        observations = super().get_many(handles)
        self.dead_lock_avoidance_agent.end_step(False)
        return observations

    def get(self, handle: int = 0):
        # all values are [0,1]
        # observation[0]  : 1 path towards target (direction 0) / otherwise 0 -> path is longer or there is no path
        # observation[1]  : 1 path towards target (direction 1) / otherwise 0 -> path is longer or there is no path
        # observation[2]  : 1 path towards target (direction 2) / otherwise 0 -> path is longer or there is no path
        # observation[3]  : 1 path towards target (direction 3) / otherwise 0 -> path is longer or there is no path
        # observation[4]  : int(agent.state == TrainState.WAITING)
        # observation[5]  : int(agent.state == TrainState.READY_TO_DEPART)
        # observation[6]  : int(agent.state == TrainState.MALFUNCTION_OFF_MAP)
        # observation[7]  : int(agent.state == TrainState.MOVING)
        # observation[8]  : int(agent.state == TrainState.STOPPED)
        # observation[9]  : int(agent.state == TrainState.MALFUNCTION)
        # observation[10] : int(agent.state == TrainState.DONE)
        # observation[11] : current agent is located at a switch, where it can take a routing decision
        # observation[12] : current agent is located at a switch, could be either a switch where it can take a decision or not (i.e. only one path)
        # observation[13] : current agent is located one step before/after a switch, where it can take a routing decision
        # observation[14] : current agent is located one step before/after a switch, could be either a switch where it can take a decision or not (i.e. only one path)
        # observation[15] : 1 if there is a path (track/branch) otherwise 0 (direction 0)
        # observation[16] : 1 if there is a path (track/branch) otherwise 0 (direction 1)
        # observation[17] : 1 if there is a path (track/branch) otherwise 0 (direction 2)
        # observation[18] : 1 if there is a path (track/branch) otherwise 0 (direction 3)
        # observation[19] : If there is a path with step (direction 0) and there is a agent with opposite direction -> 1
        # observation[20] : If there is a path with step (direction 1) and there is a agent with opposite direction -> 1
        # observation[21] : If there is a path with step (direction 2) and there is a agent with opposite direction -> 1
        # observation[22] : If there is a path with step (direction 3) and there is a agent with opposite direction -> 1
        # observation[23] : If there is a path with step (direction 0) and there is a agent with same direction -> 1
        # observation[24] : If there is a path with step (direction 1) and there is a agent with same direction -> 1
        # observation[25] : If there is a path with step (direction 2) and there is a agent with same direction -> 1
        # observation[26] : If there is a path with step (direction 3) and there is a agent with same direction -> 1
        # observation[27] : If there is a path with step (direction 0) and the target is on this path -> 1
        # observation[28] : If there is a path with step (direction 1) and the target is on this path -> 1
        # observation[29] : If there is a path with step (direction 2) and the target is on this path -> 1
        # observation[30] : If there is a path with step (direction 3) and the target is on this path -> 1
        # observation[31] : If there is a path with step (direction 0) and there is another agent's target on this path -> 1
        # observation[32] : If there is a path with step (direction 1) and there is another agent's target on this path -> 1
        # observation[33] : If there is a path with step (direction 2) and there is another agent's target on this path -> 1
        # observation[34] : If there is a path with step (direction 3) and there is another agent's target on this path -> 1
        # observation[35] : int(deadlock_avoidance_agent's action == RailEnvActions.DO_NOTHING)
        # observation[36] : int(deadlock_avoidance_agent's action == RailEnvActions.MOVE_LEFT)
        # observation[37] : int(deadlock_avoidance_agent's action == RailEnvActions.MOVE_FORWARD)
        # observation[38] : int(deadlock_avoidance_agent's action == RailEnvActions.MOVE_RIGHT)
        # observation[39] : int(deadlock_avoidance_agent's action == RailEnvActions.STOP_MOVING)
        

        observation = np.zeros(self.observation_dim)
        visited = []
        agent = self.env.agents[handle]

        agent_done = False
        if TrainState.WAITING <= agent.state <= TrainState.MALFUNCTION_OFF_MAP:
            agent_virtual_position = agent.initial_position
        elif TrainState.MOVING <= agent.state <= TrainState.MALFUNCTION:
            agent_virtual_position = agent.position
        else:   # i.e. agent.state == TrainState.DONE
            agent_virtual_position = (-1, -1)
            agent_done = True
        observation[4 + agent.state] = 1

        if not agent_done:
            visited.append(agent_virtual_position)
            distance_map = self.env.distance_map.get()
            current_cell_dist = distance_map[handle,
                                             agent_virtual_position[0], agent_virtual_position[1],
                                             agent.direction]
            possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)
            orientation = agent.direction
            if fast_count_nonzero(possible_transitions) == 1:
                orientation = fast_argmax(possible_transitions)

            for dir_loop, branch_direction in enumerate([(orientation + dir_loop) % 4 for dir_loop in range(-1, 3)]):
                if possible_transitions[branch_direction]:
                    new_position = get_new_position(agent_virtual_position, branch_direction)
                    new_cell_dist = distance_map[handle,
                                                 new_position[0], new_position[1],
                                                 branch_direction]
                    if not (np.math.isinf(new_cell_dist) and np.math.isinf(current_cell_dist)):
                        observation[1 + dir_loop] = int(new_cell_dist < current_cell_dist)

                    has_opp_agent, has_same_agent, has_target, has_opp_target, v, min_dist = self._explore(handle,
                                                                                                           new_position,
                                                                                                           branch_direction,
                                                                                                           distance_map)
                    visited.append(v)

                    if not (np.math.isinf(min_dist) and np.math.isinf(current_cell_dist)):
                        observation[16 + dir_loop] = int(min_dist < current_cell_dist)
                    observation[20 + dir_loop] = has_opp_agent
                    observation[24 + dir_loop] = has_same_agent
                    observation[28 + dir_loop] = has_target
                    observation[32 + dir_loop] = has_opp_target

            agents_on_switch, \
            agents_near_to_switch, \
            agents_near_to_switch_all, \
            agents_on_switch_all = \
                self.agent_can_choose_helper.check_agent_decision(agent_virtual_position, agent.direction)

            observation[11] = int(agents_on_switch)
            observation[12] = int(agents_on_switch_all)
            observation[13] = int(agents_near_to_switch)
            observation[14] = int(agents_near_to_switch_all)

            action = self.dead_lock_avoidance_agent.act(handle, None, eps=0)
            observation[35] = action == RailEnvActions.DO_NOTHING
            observation[36] = action == RailEnvActions.MOVE_LEFT
            observation[37] = action == RailEnvActions.MOVE_FORWARD
            observation[38] = action == RailEnvActions.MOVE_RIGHT
            observation[39] = action == RailEnvActions.STOP_MOVING

        self.env.dev_obs_dict.update({handle: visited})

        observation[np.isinf(observation)] = -1
        observation[np.isnan(observation)] = -1

        return observation
