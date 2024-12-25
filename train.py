import numpy as np
import wandb
from datetime import datetime
from torch import nn
import torch
import os
import pathlib
from argparse import ArgumentParser, Namespace
import json
import yappi

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator


from utils.render import render_env
from utils.seeding import seed_everything
# from utils.persister import load_env_from_pickle
# from utils.logger import WandbLogger
from utils.recorder import RecorderWrapper
from utils.env_creator import create_train_env
from network.rail_tranformer import RailTranformer
from network.mlp import MLP
# from reinforcement_learning.ppo import PPO
from reinforcement_learning.actor_critic import ActorCritic
from env_wrapper.railenv_wrapper import RailEnvWrapper
from observation.fast_tree_obs import FastTreeObs

# from stable_baselines3 import PPO
from reinforcement_learning.ppo import PPO
from env_wrapper.skip_no_choice_wrapper import SkipNoChoiceWrapper



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env_size", help="The size of the environment to train on. Must be one of [demo, small, medium, large, huge]", default="small", type=str)
    parser.add_argument("--network_architecture", help="The network architecture to use. Must be one of [MLP, RailTransformer]", default="MLP", type=str)
    parser.add_argument("--skip_no_choice_cells", help="Whether to skip cells where the agent has no choice", action="store_true")  # TODO: remove as it's shite!
    parser.add_argument("--normalize_v_targets", help="Whether to normalize the value targets", action="store_true")
    parser.add_argument("--log_video", help="Whether to log videos of the episodes to wandb", action="store_true")
    # TODO: try with agents masking on/off
    args = parser.parse_args()

    ### OBSERVATION ###
    TREE_OBS_DEPTH = 2
    obs_builder = FastTreeObs(max_depth=TREE_OBS_DEPTH)

    ### CONFIGURATION ###
    TOT_TIMESTEPS = 2**20    # approx 1M
    ITER_TIMESTEPS = 2**10    # approx 1K
    NUM_ITERATIONS = TOT_TIMESTEPS // ITER_TIMESTEPS

    CONFIG = {
        # Environment
        "env_size": args.env_size,
        "skip_no_choice_cells": args.skip_no_choice_cells,

        # Observation
        "tree_obs_depth": TREE_OBS_DEPTH,

        # Timesteps and iterations
        "tot_timesteps": TOT_TIMESTEPS,
        "iteration_timesteps": ITER_TIMESTEPS,
        "num_iterations": NUM_ITERATIONS,

        # Network architecture
        "network_architecture": args.network_architecture,
        "state_size": obs_builder.observation_dim,
        "action_size": 4,   # we choose to ignore the "DO_NOTHING" action (since semantically superfluous), and work only with "MOVE_LEFT", "MOVE_FORWARD", "MOVE_RIGHT" and "STOP"
        "hidden_size": 128,
        "num_layers": 3,

        # Training params
        "epochs": 10,
        "batch_size": 128,
        "learning_rate": 2.5e-4,
        "kl_limit": 0.02,
        "adam_eps": 1e-5,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

        # PPO params
        "gamma": 0.999,
        "lambda_": 0.95,
        "eps_clip": 0.2,
        "entropy_bonus": 1e-2,
        "v_target": "TD-lambda",  # "TD-lambda" (for advantage + value) or "MC" (for cumulative reward)
        "normalize_v_targets": args.normalize_v_targets,    # TODO: prova con questo OFF

        # Logging
        "batch_log_frequency": 10,    # how often to log batch stats
        "log_video": args.log_video,
        "episode_video_frequency": 10,

        # Wandb
        "wandb": True,

        # Yappi profiling
        "profiling": False,
    }

    ### ENVIRONMENT ###
    env = create_train_env(CONFIG["env_size"])
    env.obs_builder = obs_builder
    env.obs_builder.set_env(env)

    # set random seed in the config
    CONFIG["seed"] = env.random_seed

    config = Namespace(**CONFIG)

    ### WANDB ###
    if config.wandb:
        wandb.login(key="14a7d0e7554bbddd13ca1a8d45472f7a95e73ca4")
        wandb.init(project="flatland-marl", name=f"{config.env_size}_{env.number_of_agents}", config=CONFIG, sync_tensorboard=True)

        wandb.define_metric("play/step")
        wandb.define_metric("play/true_episodic_reward", step_metric="play/step")
        wandb.define_metric("play/custom_episodic_reward", step_metric="play/step")
        wandb.define_metric("play/percentage_done", step_metric="play/step")
        wandb.define_metric("play/episode_length", step_metric="play/step")

        wandb.define_metric("train/batch")
        wandb.define_metric("train/loss_pi", step_metric="train/batch")
        wandb.define_metric("train/loss_v", step_metric="train/batch")
        wandb.define_metric("train/entropy", step_metric="train/batch")
        wandb.define_metric("train/learning_rate", step_metric="train/batch")
        wandb.define_metric("train/kl_div", step_metric="train/batch")

        wandb.define_metric("action/step")
        wandb.define_metric("action/masked_agent", step_metric="action/step")
        wandb.define_metric("action/left", step_metric="action/step")
        wandb.define_metric("action/forward", step_metric="action/step")
        wandb.define_metric("action/right", step_metric="action/step")
        wandb.define_metric("action/stop", step_metric="action/step")

        wandb.define_metric("timer/step")
        wandb.define_metric("timer/inference", step_metric="timer/step")
        wandb.define_metric("timer/env_step", step_metric="timer/step")
        wandb.define_metric("timer/reward", step_metric="timer/step")
        wandb.define_metric("timer/collection", step_metric="timer/step")
        wandb.define_metric("timer/train", step_metric="timer/step")
    
    seed_everything(config.seed)

    ### WRAPPERS ###
    env = RailEnvWrapper(env) # IMPORTANT: env must be wrapped in RailEnvWrapper before any other wrapper

    if config.skip_no_choice_cells:
        env = SkipNoChoiceWrapper(env)

    if config.log_video:
        env = RecorderWrapper(env, config.episode_video_frequency)

    ### NETWORK ###
    if config.network_architecture == "MLP":
        policy_network = MLP(config.state_size, config.action_size, config.hidden_size, config.num_layers)
        value_network = MLP(config.state_size, 1, config.hidden_size, config.num_layers)
        # TODO: voglio provare sia con Tanh che con ReLU, sono troppo curiosooo
    elif config.network_architecture == "RailTransformer":
        policy_network = RailTranformer(config.state_size, config.action_size, config.hidden_size, config.num_layers)
        value_network = RailTranformer(config.state_size, 1, config.hidden_size, config.num_layers)
        # TODO: voglio provare sia con Tanh che con ReLU, sono troppo curiosooo
    else:
        raise ValueError("Invalid network architecture. Must be one of [MLP, RailTransformer]")

    ### MODEL ###
    actor_critic = ActorCritic(policy_network, value_network, config)

    print(f"Device: {config.device}")
    actor_critic.to(config.device)

    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=config.learning_rate, eps=config.adam_eps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_iterations*config.epochs, eta_min=1e-6)


    ppo = PPO(actor_critic, env, config, optimizer, scheduler)

    ppo.learn()

    now = datetime.today().strftime('%Y%m%d-%H%M')
    weights_path = f"weights/{now}_policy_{config.network_architecture}_{config.env_size}_{env.number_of_agents}_steps{config.tot_timesteps}_seed{config.seed}.pt"
    ppo.save(weights_path)
    print(f"Weights saved successfully at {weights_path}!")

    # save config as a json file
    config_path = f"weights/{now}_config_{config.network_architecture}_{config.env_size}_{env.number_of_agents}_steps{config.tot_timesteps}_seed{config.seed}.json"
    CONFIG["weights_path"] = weights_path
    CONFIG["device"] = config.device.type   # convert device to string for json serialization
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=4)
    print(f"Config saved successfully at {config_path}!")

    wandb.finish()