# epsilon_greedy.py

from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from rl.algorithms.bandits.viz import plot_trial_results
from rl.environment.bandits.k_armed_bandit import KArmedTestbed
from rl.utils.general import argmax_ties_random

matplotlib.use('TkAgg')


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
        if self.initialisation == 0:
            self.q_values = np.zeros(self.num_actions)
        elif self.initialisation > 0:
            self.q_values = np.ones(self.num_actions) * self.initialisation
        else:
            raise ValueError(f"Unrecognised initialisation: {self.initialisation}")

        self.action_counts = np.zeros(self.num_actions)

    def act(self) -> int:
        """
        Select an action using the epsilon-greedy policy.

        Returns:
            int: The action selected.
        """

        if np.random.random() < self.epsilon:
            # HOMEWORK: explore by selecting a random action
            # (use np.random.randint, with the number of actions as the upper bound)
            exploratory_action = np.random.randint(0, self.num_actions)
            return exploratory_action

        else:
            # HOMEWORK: exploit by selecting the action with the highest estimated value
            # (use argmax_ties_random to break ties randomly)
            return argmax_ties_random(self.q_values)

    def simple_update(self, action: int, reward: float) -> None:
        """
        Update the action-value estimate using sample averages.

        Args:
            action (int): The action taken.
            reward (float): The reward received.
        """
        # HOMEWORK: Increment N(A) for the selected action (c.f. self.action_counts).
        self.action_counts[action] += 1

        # HOMEWORK: Update self.q_values[action] using the incremental formula for sample averages.
        self.q_values[action] += (1 / self.action_counts[action]) * (reward - self.q_values[action])

    def weighted_update(self, action: int, reward: float) -> None:
        """
        Update the action-value estimate using a constant step size.

        Args:
            action (int): The action taken.
            reward (float): The reward received.
        """
        # HOMEWORK:
        # Update self.q_values[action] using the weighted average formula with step size alpha.
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
    num_runs = 200  # Adjust as needed (e.g., 2000 for the final version)
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
    initial_val_experiment()

    # Run showing individual runs
    # initial_val_experiment(show_individual_runs=True)

    # Run showing confidence intervals
    # initial_val_experiment(show_confidence_interval=True)

    # Run showing individual runs and confidence intervals
    # initial_val_experiment(show_individual_runs=True, show_confidence_interval=True)
