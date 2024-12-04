from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_env import RailEnv
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_count_nonzero
from flatland.envs.agent_utils import TrainState
from collections import defaultdict
import copy



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
    return list(set(decision_cells))


class SkipNoChoiceWrapper:
    def __init__(self, env: RailEnv, accumulate_skipped_rewards=True):
        self.env = env

        # TODO: remove
        # self._switches = None
        # self._switches_neighbors = None
        self._decision_cells = None
        self._accumulate_skipped_rewards = accumulate_skipped_rewards
        self._skipped_rewards = defaultdict(int)
    
    def reset(self, random_seed=None):
        obs = self.env.reset(random_seed=random_seed)
        # self._switches, self._switches_neighbors, self._decision_cells = find_all_cells_where_agent_can_choose(self.env)  # TODO: remove
        self._decision_cells = find_all_cells_where_agent_can_choose(self.env)
        return obs
    
    def step(self, action_dict):
        while True:
            prev_done = copy.deepcopy(self.env.dones)
            obs, reward, done, info = self.env.step(action_dict)

            for agent_id in obs.keys():
                # N.B. "((not prev_done[agent_id]) and done[agent_id])" fa in modo che nel buffer ci sono delle obs che servono solo a far vedere che il treno è arrivato
                # TODO IMP: prova rimuovendo "((not prev_done[agent_id]) and done[agent_id])", cioè non permettendogli di aggiornare quando ESATTAMENTE un treno
                # arriva, ma semplicemente vedrà che sarà arrivato alla prossima obs che gli darò (cioè quando sarà necessaria una azione)
                if ((not prev_done[agent_id]) and done[agent_id]) or self._on_decision_cell(self.env.agents[agent_id]):
                    if self._accumulate_skipped_rewards:
                        for handle in reward.keys():
                            reward[handle] = reward[handle] + self._skipped_rewards[handle]
                            self._skipped_rewards[handle] = 0
                    return obs, reward, done, info
                elif self._accumulate_skipped_rewards:
                    self._skipped_rewards[agent_id] += reward[agent_id]
                
            if done['__all__']:
                if self._accumulate_skipped_rewards:
                    for handle in reward.keys():
                        reward[handle] = reward[handle] + self._skipped_rewards[handle]
                        self._skipped_rewards[handle] = 0
                return obs, reward, done, info
            
            # make agents take the default, DO_NOTHING, move at next step
            action_dict = {}


    def __getattr__(self, name):
        return getattr(self.env, name)

    def _on_decision_cell(self, agent: EnvAgent):
        return (agent.position == agent.initial_position and agent.state == TrainState.READY_TO_DEPART) \
               or agent.position in self._decision_cells
            #    or agent.position is None  # removed because I don't think an agent that is done should be considered as being on a decision cell! TODO: remove

    # TODO: remove
    # def _on_switch(self, agent: EnvAgent):
    #     return agent.position in self._switches

    # def _next_to_switch(self, agent: EnvAgent):
    #     return agent.position in self._switches_neighbors
