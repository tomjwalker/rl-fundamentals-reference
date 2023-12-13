# TODO: refactor for streamlined script
#    - Agent and environment instantiated separately, in a run() function / script?
#    - `.train` method streamlined, better metric logging etc.
#    - Seeding, for reproducibility?


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from environment.k_armed_bandit import KArmedTestbed


class EpsilonGreedy:

    def __init__(self, env, epsilon, max_steps=1000, initialisation="zeros"):

        self.env = env
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.num_actions = env.bandits[0].k    # Number of bandits in any of the k-armed bandits within the testbed

        # Initialise the q-values and action counts. This reset function will be used after each k-armed bandit run.
        self.reset(initialisation="zeros")

    def _init_optimistic(self, initial_value=5):
        """
        Initialise the q-values optimistically, to encourage exploration.
        """
        return np.ones(self.num_actions) * initial_value

    def reset(self, initialisation="zeros"):
        """
        Reset the agent, by re-initialising the q-values and action counts.

        Args:
            initialisation (str): The initialisation method to use for the q-values.
        """
        # Initialise the q-values
        if initialisation == "zeros":
            self.q_values = np.zeros(self.num_actions)
        elif initialisation == "optimistic":
            self.q_values = self._init_optimistic(initial_value=5)
        else:
            raise ValueError(f"Unrecognised initialisation: {initialisation}")

        # Initialise the action counts (N)
        self.action_counts = np.zeros(self.num_actions)

    def get_argmax(self, q_values):
        """
        Get the argmax of the q-values. Splits ties randomly.

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
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return np.random.choice(ties)

    def get_action(self):
        """
        Get an action from the epsilon-greedy policy, as the argmax of the q-values with probability 1 - epsilon, and
        a random action with probability epsilon.


        Returns:
            int: The action to take.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            return self.get_argmax(self.q_values)

    def train(self):
        """
        Train the agent.
        """
        rewards_testbed = {}
        optimal_action_testbed = {}

        for k_armed_bandit in self.env.bandits:
            # TODO: set max steps in agent, or env?
            self.reset(initialisation="zeros")
            rewards = np.zeros(self.max_steps)
            optimal_action = [False] * self.max_steps
            for step in range(self.max_steps):
                action = self.get_action()
                reward = k_armed_bandit.step(action)
                self.action_counts[action] += 1
                self.q_values[action] += (1 / self.action_counts[action]) * (reward - self.q_values[action])
                rewards[step] = reward
                if action == k_armed_bandit.best_action:
                    optimal_action[step] = True
            rewards_testbed[k_armed_bandit] = rewards
            optimal_action_testbed[k_armed_bandit] = optimal_action
        # Transform from dict of arrays to pandas dataframe
        rewards_testbed = pd.DataFrame(rewards_testbed)
        optimal_action_testbed = pd.DataFrame(optimal_action_testbed)
        return rewards_testbed, optimal_action_testbed


def main():
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
        std_rewards = rewards_testbed.std(axis=1)
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


if __name__ == "__main__":
    main()

