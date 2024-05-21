import os

import numpy as np
np.random.seed(42)


def set_filepath(filepath):
    """
    Set the filepath. If any directories do not currently exist in the filepath (which may be nested, e.g. /a/b/c/),
    create them.
    """
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    return filepath


def argmax(q_values):
    """
    Gets the argmax of the q-values. Splits ties randomly.

    Args:
        q_values (np.ndarray): The q-values for each action.

    Returns:
        int: The action to take.
    """
    top = float("-inf")
    ties = []

    for i in range(len(q_values)):
        if q_values[i] > top:
            top = q_values[i]
            ties = [i]

        elif q_values[i] == top:
            ties.append(i)

    return np.random.choice(ties)
