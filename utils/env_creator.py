import pandas as pd
import pathlib

from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator

eval_list = [
    "n_agents",
    "x_dim",
    "y_dim",
    "n_cities",
    "max_rail_pairs_in_city",
    "grid_mode",
    "max_rails_between_cities",
    "malfunction_duration_min",
    "malfunction_duration_max",
    "malfunction_interval",
    "speed_ratios",
]

def create_train_env(env_size):
    """Create a training environment with the specified size.

    The environment is created using the configurations in the train_configs.csv file.

    Args:
        env_size (str): The size of the environment to create. Must be one of [demo, mini, small, medium, large, huge]

    Returns:
        env: the RailEnv environment
    """
    path = pathlib.Path(__file__).parent.parent.absolute() / "env_configs/train_configs.csv"
    configs = pd.read_csv(path, index_col=0)
    configs[eval_list] = configs[eval_list].applymap(lambda x: eval(str(x)))
    env_config = configs[configs["env_size"] == env_size].iloc[0].to_dict()

    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1/env_config["malfunction_interval"],
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
        'random_seed': env_config["random_seed"]
    }

    env = RailEnv(**env_args)
    return env

def create_test_env(test_id, env_id):
    """Create a test environment from the specified test_id and env_id.

    The environment is created using the configurations in the test_configs.csv file.

    Args:
        test_id (str): The test id of the environment to create.
        env_id (str): The env id of the environment to create.

    Returns:
        env: the RailEnv environment
    """
    path = pathlib.Path(__file__).parent.parent.absolute() / "env_configs/test_configs.csv"
    configs = pd.read_csv(path, index_col=0)
    configs[eval_list] = configs[eval_list].applymap(lambda x: eval(str(x)))

    env_config = configs[(configs["test_id"] == test_id) & (configs["env_id"] == env_id)].iloc[0].to_dict()

    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1/env_config["malfunction_interval"],
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
        # NOTE: no random_seed for test environments, as the 
        # flatland-evaluator uses a specific seed (11) for the test environments TODO: dovrei passarglielo se voglio
        # testare in locale quindi? 
    }

    env = RailEnv(**env_args)
    return env