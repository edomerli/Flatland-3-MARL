# This is the file run on AIcrowd's servers as the submission.

from argparse import Namespace
import torch
import torch.nn as nn
import pathlib
import os

from flatland.evaluators.client import FlatlandRemoteClient
from flatland.evaluators.client import TimeoutException
from flatland.core.env_observation_builder import DummyObservationBuilder

from env_wrapper.test_railenv_wrapper import TestRailEnvWrapper
from observation.binary_obs import BinaryTreeObs
from observation.binary_obs_v2 import BinaryTreeObsV2
from network.mlp import MLP
from network.rail_tranformer import RailTranformer
from utils.conversions import dict_to_tensor, tensor_to_dict

os.environ["AICROWD_TESTS_FOLDER"] = "env_configs"
remote_client = FlatlandRemoteClient()

### ARGS ###
args = {
    "network_architecture": "MLP",     # either "MLP" or "RailTransformer"
    "obs_version": "v2",        # either "v1" or "v2"
}
args = Namespace(**args)

### OBSERVATION ###
if args.obs_version == "v1":
    obs_builder = BinaryTreeObs(max_depth=2)
elif args.obs_version == "v2":
    obs_builder = BinaryTreeObsV2()
else:
    raise ValueError("Invalid observation version. Must be one of ['v1', 'v2']")

### CONFIG ###
config = {
    # network
    "state_size": obs_builder.observation_dim,
    "action_size": 5,
    "hidden_size": 128,
    "num_layers": 3,

    # misc
    "normalize_v_targets": False,
}
config = Namespace(**config)

### EVALUATION LOOP ###
episode = 0
while True:
    print("==============")
    episode += 1
    print(f"[INFO] EPISODE_START : {episode}")
    # NO WAY TO CHECK service/self.evaluation_done in client

    # use a dummy_obs_builder and then assign the right one later, similar to what we did for train with the default obs 
    # (by not passing an obs_builder in create_train_env) and a similar approach
    # to https://gitlab.aicrowd.com/flatland/flatland-starter-kit/-/blob/master/baselines/run.py?ref_type=heads
    obs, info = remote_client.env_create(obs_builder_object=DummyObservationBuilder())
    if not obs:
        """
        The remote env returns False as the first obs
        when it is done evaluating all the individual episodes
        """
        print("[INFO] DONE ALL, BREAKING")
        break

    ### WRAPPERS ###
    # BUG FIX: the number of agents is not set in the env, so we need to set it manually
    remote_client.env.number_of_agents = len(remote_client.env.agents)
    # assign the right obs_builder
    remote_client.env.obs_builder = obs_builder
    remote_client.env.obs_builder.set_env(remote_client.env)
    # wrap the remote client's env 
    remote_client.env = TestRailEnvWrapper(remote_client.env)
    # reset the components of the env, i.e. the deadlock checker and the obs_builder, and "connect" them together
    remote_client.env.reset_components()
    # recompute the obs using the actual obs_builder
    obs, _, _, info = remote_client.env_step({})

    ### NETWORK ###
    if args.network_architecture == "MLP":
        policy_network = MLP(config.state_size, config.action_size, config.hidden_size, config.num_layers)
        value_network = MLP(config.state_size, 1, config.hidden_size, config.num_layers)
    elif args.network_architecture == "RailTransformer":
        policy_network = RailTranformer(config.state_size, config.action_size, config.hidden_size, config.num_layers, activation=nn.ReLU)
        value_network = RailTranformer(config.state_size, 1, config.hidden_size, config.num_layers, activation=nn.ReLU)
    else:
        raise ValueError("Invalid network architecture. Must be one of [MLP, RailTransformer]")
        
    # load the checkpoint weights depending on the number of agents
    num_agents = remote_client.env.number_of_agents
    if num_agents <= 5:
        env_size = "demo"
        # in demo training env: curriculum is not available by definition
        recipe = "scratch"
    elif num_agents <= 20:
        env_size = "mini"
        # in mini training env: scratch > curriculum for both MLP and RailTransformer
        recipe = "scratch"
    else: 
        env_size = "small"
        # in small training env: curriculum > scratch for MLP but scratch > curriculum for RailTransformer
        recipe = "curriculum" if args.network_architecture == "MLP" else "scratch"
    # load in to CPU since ai-crowd doesn't have GPUs
    device = torch.device('cpu')
    try:
        directory = list(pathlib.Path(f"weights/{recipe}").iterdir())
        # filter only files containing the "policy" or "value" keyword, the same architecture type and the requested environment size, and get the latest one
        latest_policy_checkpoint = max(filter(lambda x: f"policy_obs{args.obs_version}_{args.network_architecture}_{env_size}" in str(x), directory), key=os.path.getctime)
        policy_network.load_state_dict(torch.load(latest_policy_checkpoint, map_location=device, weights_only=True))
        print(f"Policy network checkpoint loaded successfully from {latest_policy_checkpoint}.")
    except:
        raise ValueError(f"Unable to load checkpoint! Folder: weights/{recipe}, Architecture: {args.network_architecture}")

    ### EPISODE LOOP ###
    step = 0
    while True:
        try:
            # if all agents are waiting, i.e. possibly at the start of the episode, 
            # or if all agents are either done or deadlocked, i.e. possibly at the end of the episode,
            # then DON"T use the model to get the action but do a no-op step
            if remote_client.env.all_agents_waiting() or remote_client.env.all_done_or_deadlock():
                try:
                    obs, all_rewards, done, info = remote_client.env_step({})
                except:
                    print("[ERR] DONE BUT step()_no-op CALLED")

            else:
                # get the action from the policy network
                obs = dict_to_tensor(obs).unsqueeze(0)
                action_mask = torch.tensor(list(info["action_required"].values())).unsqueeze(0)
                action = policy_network(obs).sample()
                # apply the agents_mask to the actions, so that the agents that are not required to act will have their action set to 0
                action = action * action_mask
                action_dict = tensor_to_dict(torch.squeeze(action))

                try:
                    obs, all_rewards, done, info = remote_client.env_step(action_dict)
                except:
                    print("[ERR] DONE BUT step()_real-op CALLED")

            # break
            if done['__all__']:
                print("[INFO] EPISODE_DONE : ", episode)
                print("[INFO] TOTAL_REW: ", sum(list(all_rewards.values())))
                break

        except TimeoutException as err:
            # A timeout occurs, won't get any reward for this episode :-(
            # Skip to next episode as further actions in this one will be ignored.
            # The whole evaluation will be stopped if there are 10 consecutive timeouts.
            print("[ERR] Timeout! Will skip this episode and go to the next.", err)
            break
        step += 1

print("Evaluation complete!")
print(remote_client.submit())
