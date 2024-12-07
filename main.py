import numpy as np
import wandb
from types import SimpleNamespace
from datetime import datetime
from torch import nn
import torch

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
from network.rail_tranformer import RailTranformer
# from reinforcement_learning.ppo import PPO
from reinforcement_learning.actor_critic import ActorCritic
from env_wrapper.railenv_wrapper import RailEnvWrapper
from flatland_starter_kit.fast_tree_obs import FastTreeObs
# from stable_baselines3 import PPO
from reinforcement_learning.ppo import PPO
from env_wrapper.skip_no_choice_wrapper import SkipNoChoiceWrapper

import yappi




if __name__ == '__main__':

    ### ENVIRONMENT ###
    DEMO = True

    if DEMO:
        malfunction_parameters = MalfunctionParameters(
            malfunction_rate=1/10000,
            min_duration=20,
            max_duration=50,
        )

        demo_env_args = {
            "width": 40,
            "height": 40,
            "rail_generator": sparse_rail_generator(
                max_num_cities=2,
                grid_mode=False,
                max_rails_between_cities=2,
                max_rail_pairs_in_city=2,
            ),
            "line_generator": sparse_line_generator({1.0: 1.0, 0.5: 0.0, 0.33: 0.0, 0.25: 0.0}),
            "number_of_agents": 5,
            'malfunction_generator': ParamMalfunctionGen(malfunction_parameters),
            "random_seed": 6
        }
        env = RailEnv(**demo_env_args)
    else:
        env = load_env_from_pickle("./envs_config/train_envs/small_envs_50/Level_1.pkl")

    ### OBSERVATION ###
    TREE_OBS_DEPTH = 3  # TODO: test with higher
    env.obs_builder = FastTreeObs(max_depth=TREE_OBS_DEPTH)
    env.obs_builder.set_env(env)

    ### CONFIGURATION ###
    TOT_TIMESTEPS = 2**18    #2**21  # approx 2M
    ITER_TIMESTEPS = 2**8    #2**10  # approx 1K
    NUM_ITERATIONS = TOT_TIMESTEPS // ITER_TIMESTEPS

    CONFIG = {
        # Environment
        "env_size": "small",    # must be one of {"small", "medium", "large", "huge"}
        # TODO: connect env_size to env pickle file and to loading the respective pretrained model! All automatically
        # i.e. without having to specify the filenames etc.
        "seed": env.random_seed,
        "skip_no_choice_steps": False,  # TODO: reintroduci

        # Observation
        "tree_obs_depth": TREE_OBS_DEPTH,

        # Timesteps and iterations
        "tot_timesteps": TOT_TIMESTEPS,
        "iteration_timesteps": ITER_TIMESTEPS,
        "num_iterations": NUM_ITERATIONS,

        # Network architecture
        "model": "RailTransformer",  # "RailTransformer" or "MLP"   # TODO: implement MLP baseline or remove
        "state_size": env.obs_builder.observation_dim,
        "action_size": 4,
        "hidden_size": 256,
        "num_layers": 4,

        # Training params
        "epochs": 3,
        "batch_size": 128,  # 2**7
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

    ### WANDB ###
    if CONFIG["wandb"]:
        wandb.login(key="14a7d0e7554bbddd13ca1a8d45472f7a95e73ca4")
        wandb.init(project="flatland-marl", name=f"{CONFIG['env_size']}", config=CONFIG, sync_tensorboard=True)
        config = wandb.config

        wandb.define_metric("play/step")
        wandb.define_metric("train/batch")

        wandb.define_metric("play/episodic_reward", step_metric="play/step")
        wandb.define_metric("play/episode_length", step_metric="play/step")
        wandb.define_metric("train/loss_pi", step_metric="train/batch")
        wandb.define_metric("train/loss_v", step_metric="train/batch")
        wandb.define_metric("train/entropy", step_metric="train/batch")
        wandb.define_metric("train/lr_policy", step_metric="train/batch")
        wandb.define_metric("train/lr_value", step_metric="train/batch")
        wandb.define_metric("test/episodic_reward", step_metric="play/step")
        wandb.define_metric("test/episode_length", step_metric="play/step")
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
        policy_network = nn.Sequential(
            nn.Linear(config.state_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.action_size),
            nn.Tanh()
        )
        value_network = nn.Sequential(
            nn.Linear(config.state_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1)
        )

    ### MODEL ###
    actor_critic = ActorCritic(policy_network, value_network, config)

    print(f"Device: {config.device}")
    actor_critic.to(config.device)

    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=config.learning_rate, eps=config.adam_eps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_iterations*config.epochs, eta_min=1e-6)


    ppo = PPO(actor_critic, env, config, optimizer, scheduler)

    ppo.learn()

    now = datetime.today().strftime('%Y%m%d-%H%M')
    ppo.save(f"{now}_policy_flatland_{config.env_size}_{config.tot_timesteps}_{config.seed}.pt")

    # model = PPO(MlpPolicy, 
    #             env, 
    #             learning_rate=config.lr_policy_network, 
    #             n_steps=config.iteration_timesteps,
    #             batch_size=config.batch_size, 
    #             n_epochs=config.epochs, 
    #             gamma=config.gamma, 
    #             gae_lambda=config.lambda_,
    #             clip_range=config.eps_clip, 
    #             normalize_advantage=True, 
    #             ent_coef=config.entropy_bonus,
    #             # max_grad_norm=0.9, # default=0.5
    #             verbose=3, 
    #             seed=config.seed)
    
    # logger = WandbLogger()
    # model.set_logger(logger=logger)

    # TODO: try wandb code below, I think for histograms
    # wandb.watch(model.policy.action_net, log='all', log_freq = 1)
    # wandb.watch(model.policy.value_net, log='all', log_freq = 1)
    # collect rollouts AND train on them

    # validate performance
    # TODO: vedi Procgen's test/eval function

    wandb.finish()