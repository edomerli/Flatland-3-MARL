from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import TrainState
from collections import defaultdict
import copy


# TODO: correct it and test with it
class SkipNoChoiceWrapper:
    def __init__(self, env: RailEnv, accumulate_skipped_rewards=True):
        self.env = env

        self._accumulate_skipped_rewards = accumulate_skipped_rewards
        self._skipped_rewards = defaultdict(int)
    
    def step(self, action_dict):
        while True:
            prev_done = copy.deepcopy(self.env.dones)
            obs, reward, done, info = self.env.step(action_dict)

            for agent_id in reward.keys():
                # N.B. "((not prev_done[agent_id]) and done[agent_id])" fa in modo che nel buffer ci sono delle obs che servono solo a far vedere che il treno è arrivato
                # TODO IMP: prova rimuovendo "((not prev_done[agent_id]) and done[agent_id])", cioè non permettendogli di aggiornare quando ESATTAMENTE un treno
                # arriva, ma semplicemente vedrà che sarà arrivato alla prossima obs che gli darò (cioè quando sarà necessaria una azione)
                if ((not prev_done[agent_id]) and done[agent_id]) or info["action_required"][agent_id]:
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