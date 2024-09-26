import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')


def get_env(type):

    if type == "deterministic":
        dynamics_table = pd.DataFrame(
            data={
                "state": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                "action": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                "next_state": [0, 1, 2, 0, 1, 1, 3, 0, 0, 3, 2, 2],
                "reward": [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                "probability": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )

    elif type == "stochastic":

        dynamics_table_0 = pd.DataFrame(
            data={
                "state": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
                "action": [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, ],
                "next_state": [0, 1, 0, 1, 2, 1, 2, 0, 1, 0, ],
                "reward": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
                "probability": [2/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 2/3, ],
            }
        )
        dynamics_table_1 = pd.DataFrame(
            data={
                "state": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],
                "action": [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, ],
                "next_state": [0, 1, 1, 3, 1, 3, 0, 3, 0, 1, ],
                "reward": [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, ],
                "probability": [1/3, 2/3, 2/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, ],
            }
        )
        dynamics_table_2 = pd.DataFrame(
            data={
                "state": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, ],
                "action": [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, ],
                "next_state": [0, 3, 2, 0, 3, 2, 3, 2, 2, 0, ],
                "reward": [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, ],
                "probability": [1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 2/3, 2/3, 1/3, ],
            }
        )

        dynamics_table = pd.concat([dynamics_table_0, dynamics_table_1, dynamics_table_2])

    else:
        raise ValueError(f"Unrecognised type: {type}")

    return dynamics_table


class MCControl:

    def __init__(self, environment, epsilon, gamma):
        self.environment = environment
        self.epsilon = epsilon
        self.gamma = gamma
        self.reset

    def reset(self):

        # Initialise arbitrary policy with non-zero coverage of S, A space (go equipotential)
        state_space_size = pd.unique()
        self.policy = np.array()
