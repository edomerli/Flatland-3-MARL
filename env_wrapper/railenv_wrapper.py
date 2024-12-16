from typing import Dict

from flatland.envs.agent_utils import TrainState
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import EnvAgent
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_count_nonzero

from utils.conversions import dict_to_tensor

def find_all_cells_where_agent_can_choose(rail_env: RailEnv):
    switches = []
    switches_neighbors = []
    directions = list(range(4))
    for h in range(rail_env.height):
        for w in range(rail_env.width):
            pos = (w, h)
            is_switch = False
            # Check for switch: if there is more than one outgoing transition
            for orientation in directions:
                possible_transitions = rail_env.rail.get_transitions(*pos, orientation)
                num_transitions = fast_count_nonzero(possible_transitions)
                if num_transitions > 1:
                    switches.append(pos)
                    is_switch = True
                    break
            if is_switch:
                # Add all neighbouring rails, if pos is a switch
                for orientation in directions:
                    possible_transitions = rail_env.rail.get_transitions(*pos, orientation)
                    for movement in directions:
                        if possible_transitions[movement]:
                            switches_neighbors.append(get_new_position(pos, movement))

    # decision cells are switches and their neighbors!
    decision_cells = switches + switches_neighbors  
    # return tuple(map(set, (switches, switches_neighbors, decision_cells)))    # TODO: remove
    return set(decision_cells)

class RailEnvWrapper:
    def __init__(self, env: RailEnv):
        self.env = env

        self._decision_cells = None

        # TODO: remove?
        # self.departed_agents = set()
        # self.arrived_agents = set()
        self.deadlock_agents = set()

    # TODO: remove? lo potrei fare se volessi tenermi gli agenti (partiti e) done, a fini di passarli a NN o custom_reward
    # def step(self, action_dict_: Dict[int, RailEnvActions]):
    #     obs, rews, dones, infos = self.env.step(action_dict_)

    #     self.n_departed_agents_old = len(self.departed_agents)
    #     self.n_arrived_agents_old = len(self.arrived_agents)

    #     self.departed_agents = self.departed_agents.union({i for i, info in infos.items() if info["state"] >= TrainState.MOVING})
    #     self.arrived_agents = self.arrived_agents.union({i for i, info in infos.items() if info["state"] == TrainState.DONE})

    #     return obs, rews, dones, infos
        

    # TODO: attivalo solo se ne hai bisogno, ma faccio già che prima di ogni step, se non c'è nessun agente pronto a partire, cicla fino a quando non ce n'è almeno uno
    # def reset(self, cycle_until_action_required=True):
    #     print("Usato custom reset")
    #     obs_dict, info_dict = self.env.reset()

    #     if cycle_until_action_required:
    #         obs_dict, info_dict = self.cycle_until_action_required(info_dict)

    #     return obs_dict, info_dict

    # TODO: remove        
    # def reset(self, **kwargs):
    #     obs, info = self.env.reset(**kwargs)
    #     self.env.obs_builder.dead_lock_avoidance_agent.reset(self.env)
    #     return obs, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._decision_cells = find_all_cells_where_agent_can_choose(self.env)

        # cycle until at least one agent is ready to depart
        while not any(info["action_required"].values()):
            obs, _, _, info = self.env.step({})  # empty dict === RailEnv will choose DO_NOTHING action by default
        return obs, info
    
    def step(self, action_dict: Dict[int, RailEnvActions]):
        obs, reward, done, info = self.env.step(action_dict)

        # override action_required info
        for agent_id in reward.keys():
            agent = self.env.agents[agent_id]
            # agents that require action are those that 1) are entering their cell and 2) are either (i) moving and are on a decision cell or (ii) are ready to depart
            info["action_required"][agent_id] = info["action_required"][agent_id] \
                and (((TrainState.MOVING <= agent.state <= TrainState.STOPPED) and self._on_decision_cell(agent)) \
                        or agent.state == TrainState.READY_TO_DEPART)
            
        return obs, reward, done, info

    def _on_decision_cell(self, agent: EnvAgent):
        return (agent.position == agent.initial_position and agent.state == TrainState.READY_TO_DEPART) \
               or agent.position in self._decision_cells
            #    or agent.position is None  # removed because I don't think an agent that is done should be considered as being on a decision cell! TODO: remove



    def __getattr__(self, name):
        return getattr(self.env, name)

    # TODO: remove
    # def cycle_until_action_required(self, obs, info_dict):
    #     obs_dict = None
    #     while not any(info_dict["action_required"].values()):
    #         obs_dict, _, _, info_dict = self.env.step({})   # empty dict === RailEnv will choose DO_NOTHING action by

    #     if obs_dict is not None:
    #         obs = dict_to_tensor(obs_dict)
    #     return obs, info_dict

    def custom_reward(self, done, reward, old_info, info):
        normalized_reward = self.normalized_reward(done, reward)
        
        newly_departed_agents = 0
        newly_arrived_agents = 0
        for handle in info["state"].keys():
            if old_info["state"][handle] < TrainState.MOVING and info["state"][handle] >= TrainState.MOVING:
                newly_departed_agents += 1
            if old_info["state"][handle] < TrainState.DONE and info["state"][handle] == TrainState.DONE:
                newly_arrived_agents += 1

        # count NEW(!) deadlocks
        # TODO: remove
        # deadlock_avoidance_agent = self.env.obs_builder.dead_lock_avoidance_agent
        # agent_can_move = deadlock_avoidance_agent.agent_can_move

        _, properties, _ = self.env.obs_builder.get_properties()
        deadlocked_agents = properties["deadlocked"]

        new_deadlocks = 0
        for handle, state in info["state"].items():
            if handle in self.deadlock_agents:
                continue
            elif (TrainState.MOVING <= state <= TrainState.MALFUNCTION) and bool(deadlocked_agents[handle]): # TODO: remove -> (agent_can_move.get(handle, None) is None):
                new_deadlocks += 1
                self.deadlock_agents.add(handle)

        N = self.env.get_num_agents()

        return normalized_reward + 0.1 * (newly_departed_agents / N) + 5 * (newly_arrived_agents / N) - 2.5 * (new_deadlocks / N)

        
    def normalized_reward(self, done, reward):
        if self.is_done(done, self.env.get_info_dict()):
            if not done["__all__"]:
                # if not all agents are done, but the assert above passed, it means that some agents are done and some are in deadlock
                # -> we make the episode finish earlier by setting elapsed steps to max_episode and use it to compute the end of episode reward.
                # This way we don't pollute the replay buffer with useless transitions
                self.env._elapsed_steps = self.env._max_episode_steps
                for agent in self.env.agents:
                    reward[agent.handle] = self.env._handle_end_reward(agent)

        # normalized_reward = 1 + sum_of_agents_reward / (num_agents * T_max)
        return 1.0 + sum(reward.values()) / self.env.get_num_agents() / self.env._max_episode_steps   

    def is_done(self, done, info):
        if done["__all__"]:
            return True
        
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

