from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.step_utils.states import TrainState
from flatland.envs.fast_methods import fast_count_nonzero

from flatland_starter_kit.agent_action_config import map_rail_env_action
from flatland_starter_kit.shortest_distance_walker import ShortestDistanceWalker


class DeadlockAvoidanceObservation(DummyObservationBuilder):
    def __init__(self):
        self.counter = 0

    def get_many(self, handles: Optional[List[int]] = None) -> bool:
        self.counter += 1
        obs = np.ones(len(handles), 2)
        for handle in handles:
            obs[handle][0] = handle
            obs[handle][1] = self.counter
        return obs


class DeadlockAvoidanceShortestDistanceWalker(ShortestDistanceWalker):
    def __init__(self, env: RailEnv, agent_positions, switches):
        super().__init__(env)
        self.shortest_distance_agent_map = np.zeros((self.env.get_num_agents(),
                                                     self.env.height,
                                                     self.env.width),
                                                    dtype=int) - 1

        self.full_shortest_distance_agent_map = np.zeros((self.env.get_num_agents(),
                                                          self.env.height,
                                                          self.env.width),
                                                         dtype=int) - 1

        self.agent_positions = agent_positions

        self.opp_agent_map = {}
        self.same_agent_map = {}
        self.switches = switches

    def getData(self):
        return self.shortest_distance_agent_map, self.full_shortest_distance_agent_map

    def callback(self, handle, agent, position, direction, action, possible_transitions):
        opp_a = self.agent_positions[position]
        if opp_a != -1 and opp_a != handle:
            if self.env.agents[opp_a].direction != direction:
                d = self.opp_agent_map.get(handle, [])
                if opp_a not in d:
                    d.append(opp_a)
                self.opp_agent_map.update({handle: d})
            else:
                if len(self.opp_agent_map.get(handle, [])) == 0:
                    d = self.same_agent_map.get(handle, [])
                    if opp_a not in d:
                        d.append(opp_a)
                    self.same_agent_map.update({handle: d})

        if len(self.opp_agent_map.get(handle, [])) == 0:
            if self.switches.get(position, None) is None:
                self.shortest_distance_agent_map[(handle, position[0], position[1])] = 1
        self.full_shortest_distance_agent_map[(handle, position[0], position[1])] = 1

class DeadLockAvoidanceAgent():
    def __init__(self, env: RailEnv, action_size, enable_eps=False, show_debug_plot=False):
        # print(">> DeadLockAvoidance")
        self.env = env
        self.memory = []
        self.loss = 0
        self.action_size = action_size
        self.agent_can_move = {}
        self.agent_can_move_value = {}
        self.switches = {}
        self.show_debug_plot = show_debug_plot
        self.enable_eps = enable_eps

    def step(self, handle, state, action, reward, next_state, done):
        pass

    def act(self, handle, state, eps=0.):
        # Epsilon-greedy action selection
        if self.enable_eps:
            if np.random.random() < eps:
                return np.random.choice(np.arange(self.action_size))

        # agent = self.env.agents[state[0]]
        check = self.agent_can_move.get(handle, None)
        act = RailEnvActions.STOP_MOVING
        if check is not None:
            act = check[3]
        return map_rail_env_action(act)

    def get_agent_can_move_value(self, handle):
        return self.agent_can_move_value.get(handle, np.inf)

    def reset(self, env):
        self.env = env
        self.agent_positions = None
        self.shortest_distance_walker = None
        self.switches = {}
        for h in range(self.env.height):
            for w in range(self.env.width):
                pos = (h, w)
                for dir in range(4):
                    possible_transitions = self.env.rail.get_transitions(*pos, dir)
                    num_transitions = fast_count_nonzero(possible_transitions)
                    if num_transitions > 1:
                        if pos not in self.switches.keys():
                            self.switches.update({pos: [dir]})
                        else:
                            self.switches[pos].append(dir)

    def start_step(self, train):
        self.build_agent_position_map()
        self.shortest_distance_mapper()
        self.extract_agent_can_move()

    def end_step(self, train):
        pass

    def get_actions(self):
        pass

    def build_agent_position_map(self):
        # build map with agent positions (only agents on map)
        self.agent_positions = np.zeros((self.env.height, self.env.width), dtype=int) - 1
        for handle in range(self.env.get_num_agents()):
            agent = self.env.agents[handle]
            if TrainState.MOVING <= agent.state <= TrainState.MALFUNCTION:
                if agent.position is not None:
                    self.agent_positions[agent.position] = handle

    def shortest_distance_mapper(self):
        self.shortest_distance_walker = DeadlockAvoidanceShortestDistanceWalker(self.env,
                                                                                self.agent_positions,
                                                                                self.switches)
        for handle in range(self.env.get_num_agents()):
            agent = self.env.agents[handle]
            if agent.state < TrainState.DONE:
                self.shortest_distance_walker.walk_to_target(handle)

    def extract_agent_can_move(self):
        self.agent_can_move = {}
        shortest_distance_agent_map, full_shortest_distance_agent_map = self.shortest_distance_walker.getData()
        for handle in range(self.env.get_num_agents()):
            agent = self.env.agents[handle]
            if agent.state < TrainState.DONE:
                next_step_ok = self.check_agent_can_move(handle,
                                                         shortest_distance_agent_map[handle],
                                                         self.shortest_distance_walker.same_agent_map.get(handle, []),
                                                         self.shortest_distance_walker.opp_agent_map.get(handle, []),
                                                         full_shortest_distance_agent_map)
                if next_step_ok:
                    next_position, next_direction, action, _ = self.shortest_distance_walker.walk_one_step(handle)
                    self.agent_can_move.update({handle: [next_position[0], next_position[1], next_direction, action]})

        if self.show_debug_plot:
            a = np.floor(np.sqrt(self.env.get_num_agents()))
            b = np.ceil(self.env.get_num_agents() / a)
            for handle in range(self.env.get_num_agents()):
                plt.subplot(a, b, handle + 1)
                plt.imshow(full_shortest_distance_agent_map[handle] + shortest_distance_agent_map[handle])
            plt.show(block=False)
            plt.pause(0.01)

    def check_agent_can_move(self,
                             handle,
                             my_shortest_walking_path,
                             same_agents,
                             opp_agents,
                             full_shortest_distance_agent_map):
        agent_positions_map = (self.agent_positions > -1).astype(int)
        delta = my_shortest_walking_path
        next_step_ok = True
        for opp_a in opp_agents:
            opp = full_shortest_distance_agent_map[opp_a]
            delta = ((my_shortest_walking_path - opp - agent_positions_map) > 0).astype(int)
            if np.sum(delta) < (3 + len(opp_agents)):
                next_step_ok = False
            v = self.agent_can_move_value.get(handle, np.inf)
            v = min(v, np.sum(delta))
            self.agent_can_move_value.update({handle: v})
        return next_step_ok

    def save(self, filename):
        pass

    def load(self, filename):
        pass
