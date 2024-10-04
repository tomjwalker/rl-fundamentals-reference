"""

"""
# TODO: - Figure out why plots not exactly the same as Sutton and Barto

import numpy as np

from rl.algorithms.monte_carlo.viz import plot_results
from rl.algorithms.common.mc_agent import MonteCarloAgent
from rl.common.policy import DeterministicPolicy
from rl.common.results_logger import ResultsLogger

import gymnasium as gym
from typing import Union
from gymnasium import Env

import matplotlib.pyplot as plt


class MCExploringStartsAgent(MonteCarloAgent):

    def __init__(
            self,
            env: Union[Env, object],
            gamma: float,
            epsilon: float = None,
            logger: ResultsLogger = None,
            random_seed: int = None,
    ):
        super().__init__(env, gamma, epsilon, logger, random_seed)

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            env.reset(seed=random_seed)

        # Initialise Monte Carlo-specific attributes
        self.name = "MC Exploring Starts"  # For plotting
        self.policy = None
        self.reset()

    def _init_policy(self, state_shape, initialisation_strategy=None):
        """
        Initialises the policy for the agent.
        :param state_shape: The shape of the environment state space.
        :param initialisation_strategy: Optional strategy to initialise the policy (e.g., always hit, always stick).
        """
        self.policy = DeterministicPolicy(state_shape)

        if initialisation_strategy == "sutton_and_barto":
            # Overwrite with initial policy from Sutton and Barto
            self.policy.value[:19, :, :] = 1
        else:
            # Default initialisation (e.g., always hit)
            self.policy.value[:, :, :] = 1

    def reset(self):
        super().reset()

        # Initialise q-values, policy, and returns
        self._init_policy(self.state_shape, initialisation_strategy="sutton_and_barto")

    def act(self, state):
        """
        Wrapper for the policy's select_action method, to fit the general agent interface.
        Inspect self.policy to see how the action is selected. N.B., this is a deterministic policy (see __init__).
        """
        # HOMEWORK: use a method from the policy to select an action
        # Tip: Use `self.policy.select_action(state)` to select an action based on the current policy.
        action = self.policy.select_action(state)

        return action

    def learn(self, num_episodes=10000):
        """
        Learn the optimal policy using Monte Carlo Exploring Starts.
        """

        for episode_idx in range(num_episodes):

            # Print progress
            if episode_idx % 1000 == 0:
                print(f"Episode {episode_idx}/{num_episodes}")

            # HOMEWORK: Generate an episode.
            # There is a helper method for this within the superclass. Make sure to use exploring starts.
            episode = self._generate_episode(exploring_starts=True)

            # Loop through the episode in reverse order, updating the q-values and policy
            returns = 0
            for t, (state, action, reward) in enumerate(reversed(episode)):

                # HOMEWORK: G <- gamma * G + R_t+1
                returns = self.gamma * returns + reward

                # If the S_t, A_t pair has been seen before, continue to next for loop iteration
                # (this bit enacts first-visit MC)
                if self._is_subelement_present((state, action), episode[:len(episode) - t - 1]):
                    continue

                # Update the q-value for this state-action pair
                # NewEstimate <- OldEstimate + 1/N(St, At) * (Return - OldEstimate)
                # We'll do this in steps

                # HOMEWORK: Calculate the monte carlo error, G - Q(S_t, A_t).
                # Use the q_values dictionary to get the current estimate.
                mc_error = returns - self.q_values.get(state, action)

                # HOMEWORK: Increment N(St, At).
                # self.state_action_counts tracks N(St, At), and has a method to update it.
                self.state_action_counts.update(state, action)

                # HOMEWORK: step_size = 1 / N(St, At)
                step_size = 1 / self.state_action_counts.get(state, action)

                # HOMEWORK: Calculate NewEstimate = OldEstimate + 1/N(St, At) * (Return - OldEstimate)
                new_value = self.q_values.get(state, action) + step_size * mc_error

                # Update action value array with new estimate
                self.q_values.update(state, action, new_value)

                # Update the policy
                self.policy.update(state, self.q_values)

            # Log the episode
            self.logger.log_episode()


def smooth(x, window=1000):
    return np.convolve(x, np.ones(window), 'valid') / window


def run():

    # Set random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)

    # Run parameters
    train_episodes = 50000
    # Uncomment the next line, and run, if you want smoother plots similar to those in the lecture slides
    # train_episodes = 500000

    # Instantiate and learn the agent
    env = gym.make("Blackjack-v1", sab=True)  # `sab` means rules following Sutton and Barto
    env.reset(seed=random_seed)  # Set environment seed
    mc_control = MCExploringStartsAgent(env, gamma=1.0, random_seed=random_seed)
    mc_control.learn(num_episodes=train_episodes)

    # Plot the results
    plot_results(mc_control)

    # Plot total rewards
    plt.plot(smooth(mc_control.logger.total_rewards_per_episode))
    plt.show()


if __name__ == "__main__":
    run()