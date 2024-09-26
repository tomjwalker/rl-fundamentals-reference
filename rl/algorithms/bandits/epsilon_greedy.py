# TODO: refactor for streamlined script
#    - Agent and environment instantiated separately, in a run() function / script?
#    - `.learn` method streamlined, better metric logging etc.
#    - Seeding, for reproducibility?
from rl.environment.bandits.k_armed_bandit import KArmedTestbed
from rl.utils.general import argmax_ties_random

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def plot_trial_results(rewards_testbed, optimal_action_testbed, params, ax, color,
                       show_individual_runs=False, show_confidence_interval=False):
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
    def __init__(self, env, epsilon, max_steps=1000, initialisation=0, use_weighted_average=False, alpha=0.1):

        self.env = env
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.num_actions = env.bandits[0].k    # Number of bandits in any of the k-armed bandits within the testbed
        self.use_weighted_average = use_weighted_average
        self.alpha = alpha

        # Initialise the q-values and action counts. This reset function will be used after each k-armed bandit run.
        self.initialisation = initialisation
        self.q_values = None
        self.action_counts = None
        self.reset()

    def reset(self):
        """
        Reset the agent, by re-initialising the q-values and action counts.
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

    def act(self):
        """
        Get an action from the epsilon-greedy policy, as the argmax of the q-values with probability 1 - epsilon, and
        a random action with probability epsilon.


        Returns:
            int: The action to take.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            return argmax_ties_random(self.q_values)

    def compute_probs(self):
        """
        Compute the action probabilities for the epsilon-greedy policy.

        Returns:
            np.ndarray: The action probabilities.
        """
        # TODO: use this somewhere?
        probs = np.ones(self.num_actions) * self.epsilon / self.num_actions
        best_action = argmax_ties_random(self.q_values)
        probs[best_action] = 1 - self.epsilon + self.epsilon / self.num_actions

    def simple_update(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += (1 / self.action_counts[action]) * (reward - self.q_values[action])

    def weighted_update(self, action, reward):
        self.q_values[action] += self.alpha * (reward - self.q_values[action])

    def train(self):
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

        # Transform from dict of arrays to pandas dataframe
        rewards_testbed = pd.DataFrame(rewards_testbed)
        optimal_action_testbed = pd.DataFrame(optimal_action_testbed)
        return rewards_testbed, optimal_action_testbed


def epsilon_sweep_experiment():
    """
    Run the epsilon-greedy algorithm.
    """
    # Set the random seed
    random_seed = 0
    np.random.seed(random_seed)

    # Initialise the k-armed bandits
    num_runs = 100    # Final version: 2000
    k = 10
    k_mean = 0
    k_std = 1
    bandit_std = 1
    env = KArmedTestbed(num_runs, k, k_mean, k_std, bandit_std, random_seed)

    # Initialise the epsilon-greedy agent
    runs = {"green": 0, "red": 0.01, "blue": 0.1}    # {plot_colour: epsilon}
    results = {k: [] for k in runs.keys()}
    max_steps = 1000
    fig, ax = plt.subplots(2, 1)
    for plot_colour, epsilon in runs.items():
        print(f"Running epsilon-greedy with epsilon={epsilon}...")
        agent = EpsilonGreedy(env, epsilon, max_steps)

        # Train the agent
        rewards_testbed, optimal_action_testbed = agent.train()

        mean_rewards = rewards_testbed.mean(axis=1)
        # std_rewards = rewards_testbed.std(axis=1)
        results[plot_colour] = mean_rewards

        # Get % of time (over all runs) that the optimal action was taken
        optimal_action_fraction = optimal_action_testbed.mean(axis=1)

        # Plot the results
        ax[0].plot(mean_rewards, label=f"ε={epsilon}", color=plot_colour)
        ax[1].plot(optimal_action_fraction * 100, label=f"ε={epsilon}", color=plot_colour)

    # Plot the results
    ax[0].set_title("Average reward over time")
    ax[0].set_xlabel("Steps")
    ax[0].set_ylabel("Average reward")
    ax[0].legend()
    ax[1].set_title("Optimal action % over time")
    ax[1].set_xlabel("Steps")
    ax[1].set_ylabel("Optimal action %")
    ax[1].legend()
    plt.show()

    print("done!")


def initial_val_experiment(show_individual_runs=False, show_confidence_interval=False):
    """See effect of initialising optimistically with 5, vs 0"""
    # Set the random seed
    random_seed = 0
    np.random.seed(random_seed)

    # Initialise the k-armed bandits
    num_runs = 5
    # num_runs = 2000  # Final version
    k = 10
    k_mean = 0
    k_std = 1
    bandit_std = 1
    env = KArmedTestbed(num_runs, k, k_mean, k_std, bandit_std, random_seed)

    # Initialise the epsilon-greedy agent
    runs = {
        "grey": {"init": 0, "epsilon": 0.1, "use_weighted_average": True},
        "blue": {"init": 5, "epsilon": 0, "use_weighted_average": True}
    }
    results = {k: [] for k in runs.keys()}
    max_steps = 1000

    fig, ax = plt.subplots(2, 1, figsize=(12, 16))  # Increased figure size
    for plot_colour, params in runs.items():
        print(
            f"Running epsilon-greedy with initialisation={params['init']}, weighted average={params['use_weighted_average']}...")
        agent = EpsilonGreedy(env, params["epsilon"], max_steps, params["init"], params["use_weighted_average"])

        # Train the agent
        rewards_testbed, optimal_action_testbed = agent.train()

        # Plot the results
        plot_trial_results(rewards_testbed, optimal_action_testbed, params, ax, plot_colour,
                           show_individual_runs, show_confidence_interval)

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

if __name__ == "__main__":

    # Epsilon sweep experiment (select which visualisation option)
    # epsilon_sweep_experiment()

    # Optimistic initial value experiment (select which visualisation option)
    # initial_val_experiment()
    initial_val_experiment(show_individual_runs=True)
    # initial_val_experiment(show_confidence_interval=True)
    # initial_val_experiment(show_individual_runs=True, show_confidence_interval=True)
