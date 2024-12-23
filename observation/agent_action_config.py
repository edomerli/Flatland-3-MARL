from flatland.envs.rail_env import RailEnvActions

def map_rail_env_action(action):
    """Removes the STOP_MOVING action from the RailEnvActions enum and maps the other actions to a 0-based index.

    Args:
        action (_type_): _description_

    Returns:
        _type_: _description_
    """
    if action == RailEnvActions.MOVE_LEFT:
        return 0
    elif action == RailEnvActions.MOVE_FORWARD:
        return 1
    elif action == RailEnvActions.MOVE_RIGHT:
        return 2
    elif action == RailEnvActions.STOP_MOVING:
        return 3
    
    return 3
