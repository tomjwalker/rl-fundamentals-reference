from rl.algorithms.bandits.epsilon_greedy import EpsilonGreedy
from rl.environment.bandits.k_armed_bandit import KArmedTestbed
from rl.utils.general import argmax_ties_random

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


class UpperConfidenceBound(EpsilonGreedy):

    def __init__(self, env, c, max_steps=1000, initialisation=0):

        super().__init__(env, epsilon=0, max_steps=max_steps, initialisation=initialisation)
        self.c = c

    def act(self):
        """
        Get an action from the UCB policy, as the argmax of the q-values with probability 1 - epsilon, and
        a random action with probability epsilon.
        """
        # Get the action counts
        action_counts = self.action_counts

        # TODO: add this into the (epsilon) greedy algorithm?
        # If any actions have not been taken, take them
        if np.any(action_counts == 0):
            return np.random.choice(np.where(action_counts == 0)[0])

        # Otherwise, take the argmax of the q-values plus the exploration bonus
        # TODO: check this is correct (specifically, t)
        exploration_bonus = self.c * np.sqrt(np.log(np.sum(action_counts)) / action_counts)
        q_values = self.q_values + exploration_bonus
        return argmax_ties_random(q_values)


def main():
    # Set the random seed
    random_seed = 0
    np.random.seed(random_seed)

    # Initialise the k-armed bandits
    num_runs = 500    # Final version: 2000
    k = 10
    k_mean = 0
    k_std = 1
    bandit_std = 1
    env = KArmedTestbed(num_runs, k, k_mean, k_std, bandit_std, random_seed)

    runs = {
        "UCB c=2": {
            "agent": UpperConfidenceBound(env, c=2),
            "colour": "blue"
        },
        "epsilon=0.1": {
            "agent": EpsilonGreedy(env, epsilon=0.1),
            "colour": "grey"
        },
    }

    results = {k: [] for k in runs.keys()}
    fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
    for run_name, run in runs.items():

        print(f"Running {run_name}...")
        plot_colour = run["colour"]
        agent = run["agent"]

        # Train the agent
        rewards_testbed, optimal_action_testbed = agent.train()

        mean_rewards = rewards_testbed.mean(axis=1)
        # std_rewards = rewards_testbed.std(axis=1)
        results[run["colour"]] = mean_rewards

        # Get % of time (over all runs) that the optimal action was taken
        optimal_action_fraction = optimal_action_testbed.mean(axis=1)

        # Plot the results
        ax[0].plot(mean_rewards, label=run_name, color=plot_colour)
        ax[1].plot(optimal_action_fraction * 100, label=run_name, color=plot_colour)

    ax[0].set_title("Average reward")
    ax[0].set_xlabel("Steps")
    ax[0].set_ylabel("Average reward")
    ax[0].legend()
    ax[1].set_title("Optimal action %")
    ax[1].set_xlabel("Steps")
    ax[1].set_ylabel("Optimal action %")
    ax[1].legend()
    plt.show()


if __name__ == "__main__":
    main()
