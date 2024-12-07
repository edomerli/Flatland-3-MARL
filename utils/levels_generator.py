import os
import pandas as pd
from tqdm import tqdm

from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator

from utils.persister import save_env_to_pickle

eval_list = [
    "n_agents",
    "x_dim",
    "y_dim",
    "n_cities",
    "max_rail_pairs_in_city",
    "n_envs_run",
    "grid_mode",
    "max_rails_between_cities",
    "malfunction_duration_min",
    "malfunction_duration_max",
    "malfunction_interval",
    "speed_ratios",
]


def generate_levels(mode, test_id="all", env_id="all"):
    """This function generates the levels for the training and test environments.
       For the test environments, the RailEnvPersister.save function is used, as the flatland-evaluator uses a specific format for the environment pickle.
         For the training environments, the custom made save_env_to_pickle function is used, which also saves the random_seed.

    Args:
        mode (str): The mode in which the levels should be generated. Either "train" or "test".
        test_id (str, optional): The test id of the level(s) to generate. "all" means generate all levels with that test id. Defaults to "all".
        env_id (str, optional): The env id of the level(s) to generate. "all" means generate all levels with that env id. Defaults to "all".
    """
    ### CONFIG ###
    if mode == "test":
        PATH = "./envs_config/test_envs" 
        save_function = lambda env, path: RailEnvPersister.save(env, path, save_distance_maps=True)
    elif mode == "train":
        PATH = "./envs_config/train_envs"
        save_function = lambda env, path: save_env_to_pickle(env, path)
    else:
        raise ValueError("Invalid mode")

    # In the following, you can specify the test_id and env_id to generate the environments for. 
    # If you want to generate all environments, set test_id and env_id to "all".

    ### GENERATORION ###
    parameters_flatland = pd.read_csv(PATH + "/metadata.csv", index_col=0)
    if test_id != "all":
        parameters_flatland = parameters_flatland[parameters_flatland["test_id"] == test_id]
    if env_id != "all":
        parameters_flatland = parameters_flatland[parameters_flatland["env_id"] == env_id]
    parameters_flatland[eval_list] = parameters_flatland[eval_list].applymap(
        lambda x: eval(str(x))
    )

    for idx, env_config in tqdm(
        parameters_flatland.iterrows(), total=parameters_flatland.shape[0]
    ):
        env_config = env_config.to_dict()
        if not os.path.exists(os.path.join(PATH, env_config["test_id"])):
            os.mkdir(os.path.join(PATH, env_config["test_id"]))


        malfunction_parameters = MalfunctionParameters(
            malfunction_rate=1 / env_config["malfunction_interval"],
            min_duration=env_config["malfunction_duration_min"],
            max_duration=env_config["malfunction_duration_max"],
        )

        env_args = {
            'width': env_config["x_dim"],
            'height': env_config["y_dim"],
            'rail_generator': sparse_rail_generator(
                max_num_cities=env_config["n_cities"],
                grid_mode=env_config["grid_mode"],
                max_rails_between_cities=env_config["max_rails_between_cities"],
                max_rail_pairs_in_city=env_config["max_rail_pairs_in_city"],
            ),
            'line_generator': sparse_line_generator(env_config["speed_ratios"]),
            'number_of_agents': env_config["n_agents"],
            'malfunction_generator': ParamMalfunctionGen(malfunction_parameters),
        }

        if mode == "train":
            env_args["random_seed"] = env_config["random_seed"]

        env = RailEnv(**env_args)
        # env.reset()   # TODO: remove this line(?)
        level_id = env_config["env_id"]
        save_function(env, os.path.join(PATH, env_config["test_id"], f"{level_id}.pkl"))