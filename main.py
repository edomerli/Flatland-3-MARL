import numpy as np
import wandb
from types import SimpleNamespace
from datetime import datetime
from torch import nn
import torch
import os
import pathlib

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator


from utils.render import render_env
from utils.seeding import seed_everything
from utils.persister import load_env_from_pickle
from utils.logger import WandbLogger
from utils.recorder import RecorderWrapper
from utils.levels_generator import generate_levels
from network.rail_tranformer import RailTranformer
from network.mlp import MLP
# from reinforcement_learning.ppo import PPO
from reinforcement_learning.actor_critic import ActorCritic
from env_wrapper.railenv_wrapper import RailEnvWrapper
from flatland_starter_kit.fast_tree_obs import FastTreeObs
# from stable_baselines3 import PPO
from reinforcement_learning.ppo import PPO
from env_wrapper.skip_no_choice_wrapper import SkipNoChoiceWrapper

import yappi




if __name__ == "__main__":
    ### OBSERVATION ###
    TREE_OBS_DEPTH = 3  # TODO: test with higher
    obs_builder = FastTreeObs(max_depth=TREE_OBS_DEPTH)

    ### CONFIGURATION ###
    TOT_TIMESTEPS = 2**22    # approx 4M
    ITER_TIMESTEPS = 2**11    # approx 2K
    NUM_ITERATIONS = TOT_TIMESTEPS // ITER_TIMESTEPS

    CONFIG = {
        # Environment
        "test_id": "demo_env",
        "env_id": "Level_1",
        "skip_no_choice_steps": True,  # TODO: reintroduci

        # Observation
        "tree_obs_depth": TREE_OBS_DEPTH,

        # Timesteps and iterations
        "tot_timesteps": TOT_TIMESTEPS,
        "iteration_timesteps": ITER_TIMESTEPS,
        "num_iterations": NUM_ITERATIONS,

        # Network architecture
        "model": "MLP",  # "RailTransformer" or "MLP"   # TODO: implement MLP baseline or remove
        "state_size": obs_builder.observation_dim,
        "action_size": 4,
        "hidden_size": 256,
        "num_layers": 4,

        # Training params
        "epochs": 5,    # TODO: try 3, 5 and 10
        "batch_size": 2**6,
        "learning_rate": 2.5e-4,
        "kl_limit": 0.02,
        "adam_eps": 1e-5,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

        # PPO params
        "gamma": 0.999,
        "lambda_": 0.95,
        "eps_clip": 0.2,
        "entropy_bonus": 1e-5,
        "v_target": "TD-lambda",  # "TD-lambda" (for advantage + value) or "MC" (for cumulative reward)
        "normalize_v_targets": True,    # TODO: prova con questo OFF

        # Logging
        "log_frequency": 10,
        "log_video": False,
        "episode_video_frequency": 10,

        # Wandb
        "wandb": True,

        # Yappi profiling
        "profiling": False,
    }

    ### ENVIRONMENT ###
    # for the notebook version:
    # pickle_train_env_path = f"./envs_config/train_envs/{CONFIG['test_id']}/{CONFIG['env_id']}.pkl"
    # for the script version:
    pickle_train_env_path = pathlib.Path(__file__).parent.absolute() / f"envs_config/train_envs/{CONFIG['test_id']}/{CONFIG['env_id']}.pkl"
    
    # generate the level if the pickle file does not exist
    if not os.path.exists(pickle_train_env_path):
        generate_levels("train", CONFIG["test_id"], CONFIG["env_id"])

    env = load_env_from_pickle(pickle_train_env_path)

    env.obs_builder = obs_builder
    env.obs_builder.set_env(env)

    # set random seed in the config
    CONFIG["seed"] = env.random_seed

    env_size = CONFIG["test_id"].split("_")[0]


    ### WANDB ###
    if CONFIG["wandb"]:
        wandb.login(key="14a7d0e7554bbddd13ca1a8d45472f7a95e73ca4")
        wandb.init(project="flatland-marl", name=f"{env_size}_{env.number_of_agents}", config=CONFIG, sync_tensorboard=True)
        config = wandb.config

        wandb.define_metric("play/step")
        wandb.define_metric("train/batch")

        wandb.define_metric("play/true_episodic_reward", step_metric="play/step")
        wandb.define_metric("play/custom_episodic_reward", step_metric="play/step")
        wandb.define_metric("play/episode_length", step_metric="play/step")
        wandb.define_metric("play/percentage_done", step_metric="play/step")
        wandb.define_metric("train/loss_pi", step_metric="train/batch")
        wandb.define_metric("train/loss_v", step_metric="train/batch")
        wandb.define_metric("train/entropy", step_metric="train/batch")
        wandb.define_metric("train/learning_rate", step_metric="train/batch")
        wandb.define_metric("train/kl_div", step_metric="train/batch")
        wandb.define_metric("test/true_episodic_reward", step_metric="play/step")   # TODO: mettere test/step qui come step_metric? Check wandb plots
        wandb.define_metric("test/custom_episodic_reward", step_metric="play/step")
        wandb.define_metric("test/episode_length", step_metric="play/step")
        wandb.define_metric("test/percentage_done", step_metric="play/step")
    else:
        config = SimpleNamespace(**CONFIG)

    seed_everything(config.seed)

    # IMPORTANT: env must be wrapped in RailEnvWrapper before any other wrapper
    env = RailEnvWrapper(env)

    if config.skip_no_choice_steps:
        env = SkipNoChoiceWrapper(env)

    if config.log_video:
        env = RecorderWrapper(env, config.episode_video_frequency)


    # env_steps = 1000  # 2 * env.width * env.height  # Code uses 1.5 to calculate max_steps
    # rollout_fragment_length = 50
    # # env = ss.black_death_v2(env)    
    # env = ss.vector.markov_vector_wrapper.MarkovVectorEnv(env, black_death=True)    # to handle varying number of agents
    # env = ss.concat_vec_envs_v0(env, 4, num_cpus=1, base_class='stable_baselines3')

    # env.reset()
    # o, r, d, i = env.step({i: 0 for i in range(50)})
    # print(f"obs: {o}\n rewards: {r}\n dones: {d}\n infos: {i}")
    # exit()

    ### NETWORK ###
    if config.model == "RailTransformer":
        policy_network = RailTranformer(config.state_size, config.action_size, config.hidden_size, config.num_layers, activation=nn.Tanh)
        value_network = RailTranformer(config.state_size, 1, config.hidden_size, config.num_layers, activation=nn.Tanh)
        # TODO: voglio provare sia con Tanh che con ReLU, sono troppo curiosooo
    elif config.model == "MLP":
        policy_network = MLP(config.state_size, config.action_size, config.hidden_size, 3, activation=nn.Tanh)
        value_network = MLP(config.state_size, 1, config.hidden_size, 3, activation=nn.Tanh)

    ### MODEL ###
    actor_critic = ActorCritic(policy_network, value_network, config)

    print(f"Device: {config.device}")
    actor_critic.to(config.device)

    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=config.learning_rate, eps=config.adam_eps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_iterations*config.epochs, eta_min=1e-6)


    ppo = PPO(actor_critic, env, config, optimizer, scheduler)

    ppo.learn()

    now = datetime.today().strftime('%Y%m%d-%H%M')
    ppo.save(f"{now}_policy_flatland_{env_size}_{env.number_of_agents}_{config.tot_timesteps}_{config.seed}.pt")
    

    # TODO: try wandb code below, I think for histograms
    # wandb.watch(model.policy.action_net, log='all', log_freq = 1)
    # wandb.watch(model.policy.value_net, log='all', log_freq = 1)

    # validate performance
    # TODO: vedi Procgen's test/eval function

    wandb.finish()