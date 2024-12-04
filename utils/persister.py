import pickle

def save_env_to_pickle(env, path):
    """
    Save the environment to a pickle file.
    """
    f = open(path, 'wb')
    pickle.dump(env,f)
    f.close()


def load_env_from_pickle(path):
    """
    Load the environment from a pickle file.
    """
    f = open(path, 'rb')
    env = pickle.load(f)
    f.close()

    return env