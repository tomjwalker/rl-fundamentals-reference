# epsilon_greedy.py

from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

from rl.environment.bandits.k_armed_bandit import KArmedTestbed
from rl.utils.general import argmax_ties_random

matplotlib.use('TkAgg')


def plot_trial_results(
    rewards_testbed: pd.DataFrame,
    optimal_action_testbed: pd.DataFrame,
    params: Dict[str, float],
    ax: List[plt.Axes],
    color: str,
    show_individual_runs: bool = False,
    show_confidence_interval: bool = False
) -> None:
    """
    Plot the results of the trials for a given set of parameters.

    Args:
        rewards_testbed (pd.DataFrame): DataFrame containing rewards for each step and run.
        optimal_action_testbed (pd.DataFrame): DataFrame indicating if the optimal action was taken.
        params (Dict[str, float]): Parameters used in the experiment.
        ax (List[plt.Axes]): List of matplotlib axes to plot on.
        color (str): Color to use for plotting.
        show_individual_runs (bool): Whether to plot individual runs.
        show_confidence_interval (bool): Whether to show the confidence interval.
    """
    num_runs = rewards_testbed.shape[1]
    steps = rewards_testbed.shape[0]

    # Plot individual runs if specified
    alpha = 0.15
    if show_individual_runs:
        for i in range(num_runs):
            ax[0].plot(rewards_testbed.iloc[:, i], color=color, alpha=alpha)
            ax[1].plot(optimal_action_testbed.iloc[:, i] * 100, color=color, alpha=alpha)

    # Calculate mean
    mean_rewards = rewards_testbed.mean(axis=1)
    mean_optimal = optimal_action_testbed.mean(axis=1) * 100

    # Plot mean
    ax[0].plot(mean_rewards, color=color, linewidth=2, label=f"init={params['init']}")
    ax[1].plot(mean_optimal, color=color, linewidth=2, label=f"init={params['init']}")

    # Calculate and plot confidence interval if specified
    if show_confidence_interval:
        std_rewards = rewards_testbed.std(axis=1)
        ci_rewards = stats.t.interval(0.95, num_runs - 1, loc=mean_rewards, scale=std_rewards / np.sqrt(num_runs))

        std_optimal = optimal_action_testbed.std(axis=1) * 100
        ci_optimal = stats.t.interval(0.95, num_runs - 1, loc=mean_optimal, scale=std_optimal / np.sqrt(num_runs))

        ax[0].fill_between(range(steps), ci_rewards[0], ci_rewards[1], color=color, alpha=0.3)
        ax[1].fill_between(range(steps), ci_optimal[0], ci_optimal[1], color=color, alpha=0.3)


class EpsilonGreedy:
    def __init__(
        self,
        env: KArmedTestbed,
        epsilon: float,
        max_steps: int = 1000,
        initialisation: float = 0,
        use_weighted_average: bool = False,
        alpha: float = 0.1
    ) -> None:
        """
        Initialise the EpsilonGreedy agent.

        Args:
            env (KArmedTestbed): The k-armed bandit testbed environment.
            epsilon (float): The probability of choosing a random action (exploration rate).
            max_steps (int): The maximum number of steps per run.
            initialisation (float): The initial value for all action-value estimates.
            use_weighted_average (bool): Whether to use a constant step size (True) or sample averages (False).
            alpha (float): The step size parameter for weighted updates.
        """
        self.env = env
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.num_actions = env.bandits[0].k  # Number of actions
        self.use_weighted_average = use_weighted_average
        self.alpha = alpha

        self.initialisation = initialisation
        self.q_values: np.ndarray = np.zeros(self.num_actions)
        self.action_counts: np.ndarray = np.zeros(self.num_actions)
        self.reset()

    def reset(self) -> None:
        """
        Reset the agent by re-initialising the action-value estimates and action counts.
        """
        # Initialise the q-values
        if self.initialisation == 0:
            self.q_values = np.zeros(self.num_actions)
        elif self.initialisation > 0:
            self.q_values = np.ones(self.num_actions) * self.initialisation
        else:
            raise ValueError(f"Unrecognised initialisation: {self.initialisation}")

        # Initialise the action counts (N)
        self.action_counts = np.zeros(self.num_actions)

    def act(self) -> int:
        """
        Select an action using the epsilon-greedy policy.

        Returns:
            int: The action selected.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            return argmax_ties_random(self.q_values)

    def simple_update(self, action: int, reward: float) -> None:
        """
        Update the action-value estimate using sample averages.

        Args:
            action (int): The action taken.
            reward (float): The reward received.
        """
        self.action_counts[action] += 1
        self.q_values[action] += (1 / self.action_counts[action]) * (reward - self.q_values[action])

    def weighted_update(self, action: int, reward: float) -> None:
        """
        Update the action-value estimate using a constant step size.

        Args:
            action (int): The action taken.
            reward (float): The reward received.
        """
        self.q_values[action] += self.alpha * (reward - self.q_values[action])

    def train(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Train the agent over all runs in the environment.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing rewards and optimal action indicators.
        """
        rewards_testbed = {}
        optimal_action_testbed = {}

        for k_armed_bandit in self.env.bandits:
            self.reset()
            rewards = np.zeros(self.max_steps)
            optimal_action = [False] * self.max_steps
            for step in range(self.max_steps):
                action = self.act()
                reward = k_armed_bandit.step(action)

                if self.use_weighted_average:
                    self.weighted_update(action, reward)
                else:
                    self.simple_update(action, reward)

                rewards[step] = reward
                if action == k_armed_bandit.best_action:
                    optimal_action[step] = True
            rewards_testbed[k_armed_bandit] = rewards
            optimal_action_testbed[k_armed_bandit] = optimal_action

        # Transform from dict of arrays to pandas DataFrame
        rewards_testbed_df = pd.DataFrame(rewards_testbed)
        optimal_action_testbed_df = pd.DataFrame(optimal_action_testbed)
        return rewards_testbed_df, optimal_action_testbed_df


def epsilon_sweep_experiment() -> None:
    """
    Run the epsilon-greedy algorithm with different epsilon values and plot the results.
    """
    # Set the random seed for reproducibility
    random_seed = 0
    np.random.seed(random_seed)

    # Initialise the k-armed bandits
    num_runs = 100  # Final version: 2000
    k = 10
    k_mean = 0
    k_std = 1
    bandit_std = 1
    env = KArmedTestbed(num_runs, k, k_mean, k_std, bandit_std, random_seed)

    # Define the epsilon values to test
    runs = {"green": 0, "red": 0.01, "blue": 0.1}  # {plot_colour: epsilon}
    max_steps = 1000
    fig, ax = plt.subplots(2, 1)
    for plot_colour, epsilon in runs.items():
        print(f"Running epsilon-greedy with epsilon={epsilon}...")
        agent = EpsilonGreedy(env, epsilon, max_steps)

        # Train the agent
        rewards_testbed, optimal_action_testbed = agent.train()

        mean_rewards = rewards_testbed.mean(axis=1)
        optimal_action_fraction = optimal_action_testbed.mean(axis=1)

        # Plot the results
        ax[0].plot(mean_rewards, label=f"ε={epsilon}", color=plot_colour)
        ax[1].plot(optimal_action_fraction * 100, label=f"ε={epsilon}", color=plot_colour)

    # Set titles and labels
    ax[0].set_title("Average reward over time")
    ax[0].set_xlabel("Steps")
    ax[0].set_ylabel("Average reward")
    ax[0].legend()
    ax[1].set_title("Optimal action % over time")
    ax[1].set_xlabel("Steps")
    ax[1].set_ylabel("Optimal action %")
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    print("Experiment complete!")


def initial_val_experiment(
    show_individual_runs: bool = False,
    show_confidence_interval: bool = False
) -> None:
    """
    Run the optimistic initial values experiment and plot the results.

    Args:
        show_individual_runs (bool): Whether to plot individual runs.
        show_confidence_interval (bool): Whether to show the confidence interval.
    """
    # Set the random seed for reproducibility
    random_seed = 0
    np.random.seed(random_seed)

    # Initialise the k-armed bandits
    num_runs = 5  # Adjust as needed (e.g., 2000 for the final version)
    k = 10
    k_mean = 0
    k_std = 1
    bandit_std = 1
    env = KArmedTestbed(num_runs, k, k_mean, k_std, bandit_std, random_seed)

    # Define the runs with different initialisations
    runs = {
        "grey": {"init": 0, "epsilon": 0.1, "use_weighted_average": True},
        "blue": {"init": 5, "epsilon": 0, "use_weighted_average": True}
    }
    max_steps = 1000

    fig, ax = plt.subplots(2, 1, figsize=(12, 16))  # Increased figure size
    for plot_colour, params in runs.items():
        print(
            f"Running with initialisation={params['init']}, epsilon={params['epsilon']}, "
            f"weighted average={params['use_weighted_average']}..."
        )
        agent = EpsilonGreedy(
            env,
            params["epsilon"],
            max_steps,
            params["init"],
            params["use_weighted_average"]
        )

        # Train the agent
        rewards_testbed, optimal_action_testbed = agent.train()

        # Plot the results
        plot_trial_results(
            rewards_testbed,
            optimal_action_testbed,
            params,
            ax,
            plot_colour,
            show_individual_runs,
            show_confidence_interval
        )

    # Set titles and labels
    ax[0].set_title("Average reward over time")
    ax[0].set_xlabel("")  # Remove x-label from top subplot
    ax[0].set_ylabel("Average reward")
    ax[0].legend()

    ax[1].set_title("Optimal action % over time")
    ax[1].set_xlabel("Steps")
    ax[1].set_ylabel("Optimal action %")
    ax[1].legend()

    plt.tight_layout()  # Adjust the layout
    plt.subplots_adjust(hspace=0.3)  # Increase vertical space between subplots
    plt.show()

    print("Experiment complete!")


if __name__ == "__main__":
    """
    Main function to run experiments.

    To run an experiment, uncomment the corresponding function call.

    Available Experiments:
    1. Epsilon Sweep Experiment
    2. Optimistic Initial Values Experiment

    Instructions:
    - Uncomment the function call for the experiment you wish to run.
    - Only run one experiment at a time to avoid conflicts.
    """

    # Epsilon Sweep Experiment
    # epsilon_sweep_experiment()

    # Optimistic Initial Values Experiment
    # Uncomment one of the following lines to run the experiment with desired options:

    # Run with default settings
    # initial_val_experiment()

    # Run showing individual runs
    # initial_val_experiment(show_individual_runs=True)

    # Run showing confidence intervals
    # initial_val_experiment(show_confidence_interval=True)

    # Run showing individual runs and confidence intervals
    # initial_val_experiment(show_individual_runs=True, show_confidence_interval=True)
