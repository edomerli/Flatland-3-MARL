from flatland.envs.rail_env import RailEnv
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_count_nonzero

def find_switches_and_switches_neighbors(rail_env: RailEnv):
    """
    Computes all cells where the agent can choose an action.
    Decision cells are:
    1) decision_switches -> cells where an agent moving in a certain direction can choose its path. N.B. they depend on the agent's direction!
    2) switches_neighbors -> cells right before a switch with respect to a certain direction. 
                             N.B. ANY switch, even those in which the agent can't choose its path! I.e. the actions that can be choosen here are MOVE_FORWARD and STOP_MOVING.

    Args:
        rail_env (RailEnv): the environment

    Returns:
        dict: a dictionary where the keys are the directions and the values are sets of cells where agents coming from that direction can choose between more than one action.
    """

    # find all switches first
    switches = {dir: [] for dir in range(4)}
    for h in range(rail_env.height):
        for w in range(rail_env.width):
            pos = (w, h)
            for dir in range(4):
                possible_transitions = rail_env.rail.get_transitions(*pos, dir)
                num_transitions = fast_count_nonzero(possible_transitions)
                if num_transitions > 1:
                    switches[dir].append(pos)
    
    # convert switches lists to sets for efficient lookup
    switches = {dir: set(switches[dir]) for dir in switches}
    all_switches_set = set.union(*switches.values())

    # find all switches neighbors
    switches_neighbors = {dir: [] for dir in range(4)}
    for h in range(rail_env.height):
        for w in range(rail_env.width):
            pos = (w, h)
            for dir in range(4):
                possible_transitions = rail_env.rail.get_transitions(*pos, dir)
                # look one step forward in each direction
                for new_dir in range(4):
                    # check if you could move there
                    if possible_transitions[new_dir]:
                        # check if the new position is a switch and the current position is not
                        new_pos = get_new_position(pos, new_dir)
                        if new_pos in all_switches_set and pos not in all_switches_set:
                            switches_neighbors[dir].append(pos)
                            break
                    
    # convert switches neighbors lists to sets for efficient lookup
    switches_neighbors = {dir: set(switches_neighbors[dir]) for dir in switches_neighbors}

    return switches, switches_neighbors

def find_decision_cells(rail_env: RailEnv):
    """Find all cells where the agent can choose an action.

    Args:
        rail_env (RailEnv): the environment of which to find the decision cells

    Returns:
        decision_cells: a dictionary where the keys are the directions and the values are sets of cells where agents coming from that direction can choose between more than one action.
    """
    switches, switches_neighbors = find_switches_and_switches_neighbors(rail_env)
    
    # combine switches and switches_neighbors to get decision cells
    decision_cells = {dir: switches[dir].union(switches_neighbors[dir]) for dir in range(4)}

    return decision_cells