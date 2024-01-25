import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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


def get_policy(policy_type, env_type):

    if policy_type == "equiprobable":
        policy = np.array([[1 / 4, 1 / 4, 1 / 4, 1 / 4], [1 / 4, 1 / 4, 1 / 4, 1 / 4], [1 / 4, 1 / 4, 1 / 4, 1 / 4]])
    elif policy_type == "optimal":
        if env_type == "deterministic":
            policy = np.array([[0, 0.5, 0.5, 0], [0, 0, 1, 0], [0, 1, 0, 0]])
        elif env_type == "stochastic":
            policy = np.array([[0, 0.5, 0.5, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        else:
            raise ValueError(f"Unrecognised env_type: {env_type}")
    else:
        raise ValueError(f"Unrecognised policy_type: {policy_type}")

    return policy

GAMMA = 0.5


def get_dynamics(state, action):
    """
    Get the dynamics of the environment, given a state and action.

    Args:
        state (int): The current state.
        action (int): The action to take.

    Returns:
        dynamics (pandas.DataFrame): Dataframe containing all probabilistic dynamics of the next state and reward given
            the current state and action.
    """
    dynamics = DYNAMICS_TABLE[(DYNAMICS_TABLE["state"] == state) & (DYNAMICS_TABLE["action"] == action)]
    dynamics.reset_index(drop=True, inplace=True)
    return dynamics


def evaluate_policy(policy, gamma, theta=1e-10):
    """
    Evaluate a policy.

    Args:
        policy (np.ndarray): The policy to evaluate.

    Returns:
        np.ndarray: The value of the policy.
    """

    states = DYNAMICS_TABLE["state"].unique()
    actions = DYNAMICS_TABLE["action"].unique()

    # Initialise value arbitrarily
    value = np.zeros(len(states) + 1)    # Include terminal state for calculations

    log = {state: [] for state in states}

    # Iterate until convergence
    while True:
        delta = 0

        for state in states:
            old_value = value[state]
            new_value = 0
            for action in actions:
                dynamics = get_dynamics(state, action)
                new_value += policy[state, action] * np.sum(
                    dynamics["probability"] * (dynamics["reward"] + gamma * value[dynamics["next_state"]])
                )
            value[state] = new_value
            delta = max(delta, np.abs(old_value - value[state]))

            log[state].append(value[state])

        if delta < theta:
            break

    return value, log


def plot_log(log):
    """
    Figure of 3x1 subplots, with each subplot showing the value of a state over time.
    Each axis is labelled with the state number, as well as the final value of the state.
    """

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for state, ax in zip(log.keys(), axes):
        ax.plot(log[state], marker="o")
        ax.set_title(f"State {state}; value = {round(log[state][-1], 3)}; iterations = {len(log[state])}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    env_type = "deterministic"    # "deterministic" or "stochastic"
    policy_type = "equiprobable"    # "equiprobable" or "optimal"

    # Get the environment dynamics
    DYNAMICS_TABLE = get_env(env_type)

    # Get the policy
    policy = get_policy(policy_type, env_type)

    value, log = evaluate_policy(policy, GAMMA, theta=1e-10)

    print(f"Value: {value}")

    plot_log(log)
