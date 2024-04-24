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


def argmax_equiprob_ties(array):
    """
    Returns the index of the maximum value in the array. In the case of ties, weights these equiprobably.

    For example, if the array is [0, 1, 1, 0], then the function will return [0, 0.5, 0.5, 0].

    Args:
        array (np.ndarray): The array to find the argmax_equiprob_ties of.

    Returns:
        np.ndarray: The argmax_equiprob_ties of the array.
    """
    max_value = np.max(array)
    max_indices = np.where(array == max_value)[0]
    return (np.eye(len(array))[max_indices].sum(axis=0)) / len(max_indices)


def plot_log_subplot(log, state, ax=None, figsize=(10, 10)):
    """
    Creates a subplot showing the value of a state over time.

    Args:
        log (dict): A dictionary containing state values over iterations.
        state (int): The state number to plot.
        ax (matplotlib.axes.Axes, optional): An existing axis to plot on.
            If None, a new subplot is created.
        figsize (tuple, optional): Figure size if a new subplot is created.
            Defaults to (10, 3).
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)  # Create fig if necessary

    ax.plot(log[state], marker="o")
    ax.set_title(f"State {state}; value = {round(log[state][-1], 3)}; iterations = {len(log[state]) - 1}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")

    return ax


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


def initialise_artefacts(env, initial_policy="equiprobable"):
    """
    Initialise the value and policy artefacts.

    Args:
        env (pandas.DataFrame): The environment dynamics.
        initial_policy (str): The initial policy to use.

    Returns:
        dict: A dictionary containing the value and policy artefacts.
    """

    states = DYNAMICS_TABLE["state"].unique()

    # Initialise value arbitrarily
    value = np.zeros(len(states) + 1)    # Include terminal state for calculations

    # Get the policy
    policy = get_policy(initial_policy, env_type)

    return {"value": value, "policy": policy}


def evaluate_policy(policy, value, gamma, theta=1e-10):
    """
    Evaluate a policy.

    Args:
        policy (np.ndarray): The policy to evaluate.

    Returns:
        np.ndarray: The value of the policy.
    """

    states = DYNAMICS_TABLE["state"].unique()
    actions = DYNAMICS_TABLE["action"].unique()

    log = {state: [value[state]] for state in states}

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


def improve_policy(value, policy, gamma):
    """
    Improve a policy.

    Args:
        value (np.ndarray): The value of the policy.
        policy (np.ndarray): The policy to improve.

    Returns:
        policy (np.ndarray): The improved policy.
        policy_stable (bool): Whether the policy is stable. A flag to terminate the policy iteration.
    """

    states = DYNAMICS_TABLE["state"].unique()
    actions = DYNAMICS_TABLE["action"].unique()
    policy_stable = True

    for state in states:
        old_action = policy[state].copy()
        action_returns = np.zeros(len(actions))
        for action in actions:
            dynamics = get_dynamics(state, action)
            action_returns[action] = np.sum(
                dynamics["probability"] * (dynamics["reward"] + gamma * value[dynamics["next_state"]])
            )
        policy[state] = argmax_equiprob_ties(action_returns)
        # Check if policy is stable
        if not np.array_equal(old_action, policy[state]):
            policy_stable = False

    return policy, policy_stable


def iterate_policy(policy, value, gamma, theta=1e-10):
    """
    Iterate a policy until convergence.

    Args:
        policy (np.ndarray): The policy to iterate.
        value (np.ndarray): The value of the policy.
        gamma (float): The discount factor.
        theta (float): The convergence threshold.

    Returns:
        np.ndarray: The optimal policy.
    """
    policy_stable = False
    num_iterations = 0
    plots_per_row = 3  # Customize this!
    fig, axes = plt.subplots(nrows=1, ncols=plots_per_row, figsize=(15, 5))  # Initial figure layout
    axes = axes.flatten()

    while not policy_stable:
        value, log = evaluate_policy(policy, value, gamma, theta)
        policy, policy_stable = improve_policy(value, policy, gamma)

        print("=" * 20)
        print(f"Iteration: {num_iterations}")
        print(f"Value: {value}")
        print(f"Policy: {policy}")

        # Plotting Logic
        for state, ax in zip(log.keys(), axes):  # Iterate over axes
            plot_log_subplot(log, state, ax=ax)

            if (state + 1) % plots_per_row == 0 and state != 0:  # Check if it's time for a new row
                fig.suptitle(f"Policy Iteration {num_iterations}", fontsize=14)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust title position
                fig, axes = plt.subplots(nrows=1, ncols=plots_per_row, figsize=(15, 5))
                axes = axes.flatten()

        num_iterations += 1

    # Clean up after last iteration
    fig.suptitle(f"Policy Iteration {num_iterations}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust title position
    for ax in axes[len(log):]:
        ax.remove()

    return policy, fig



if __name__ == "__main__":

    GAMMA = 0.5

    env_type = "stochastic"    # "deterministic" or "stochastic"
    policy_type = "equiprobable"    # "equiprobable" or "optimal"

    # Get the environment dynamics
    DYNAMICS_TABLE = get_env(env_type)

    # Initialise the value and policy artefacts
    initialised_artefacts = initialise_artefacts(DYNAMICS_TABLE, initial_policy=policy_type)
    value = initialised_artefacts["value"]
    policy = initialised_artefacts["policy"]

    # value, log = evaluate_policy(policy, GAMMA, theta=1e-10)
    # plot_log(log)
    #
    # new_policy, policy_stable = improve_policy(value, policy, GAMMA)
    # print(new_policy)

    optimal_policy, results_figure = iterate_policy(policy, value, GAMMA)
    plt.show()
