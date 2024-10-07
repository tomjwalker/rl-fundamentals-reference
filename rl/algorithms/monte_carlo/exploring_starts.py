"""

"""
# TODO: - Figure out why plots not exactly the same as Sutton and Barto

import numpy as np

from rl.algorithms.monte_carlo.viz import plot_results
from rl.algorithms.common.mc_agent import MonteCarloAgent
from rl.common.policy import DeterministicPolicy
from rl.common.results_logger import ResultsLogger

import gymnasium as gym
from typing import Union, Tuple, List
from gymnasium import Env

import matplotlib.pyplot as plt


class MCExploringStartsAgent(MonteCarloAgent):
    """
    An implementation of the Monte Carlo Exploring Starts agent.
    This agent uses the Monte Carlo Exploring Starts algorithm to learn the optimal policy for a given environment.
    """

    def __init__(
            self,
            env: Union[Env, object],
            gamma: float,
            epsilon: float = None,
            logger: ResultsLogger = None,
            random_seed: int = None,
    ) -> None:
        """
        Initialises the Monte Carlo Exploring Starts agent.

        Args:
            env (Union[Env, object]): The environment to interact with.
            gamma (float): Discount factor.
            epsilon (float, optional): Exploration parameter (not used in this agent).
            logger (ResultsLogger, optional): Logger to record training results.
            random_seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(env, gamma, epsilon, logger, random_seed)

        # Initialise Monte Carlo-specific attributes
        self.name: str = "MC Exploring Starts"  # For plotting
        self.policy: Union[DeterministicPolicy, None] = None
        self.reset()

    def _init_policy(self, state_shape: Tuple[int, ...]) -> None:
        """
        Initialises the policy for the agent using the policy initialisation from Sutton and Barto (pp. 93).

        Args:
            state_shape (Tuple[int, ...]): The shape of the environment state space.

        Policy details:
            - If player sum == 20 or 21, stick (0)
            - Otherwise, hit (1)
        """
        # TODO: environment-specific. Refactor to be more general
        self.policy = DeterministicPolicy(state_shape)

        # Overwrite with initial policy from Sutton and Barto
        self.policy.action_map[:19, :, :] = 1

    def reset(self) -> None:
        """
        Resets the agent's attributes, including q-values, policy, and returns.
        """
        super().reset()

        # Initialise q-values, policy, and returns
        self._init_policy(self.state_shape)

    def act(self, state: Tuple[int, ...]) -> int:
        """
        Selects an action based on the current policy.

        Args:
            state (Tuple[int, ...]): The current state of the environment.

        Returns:
            int: The action selected by the policy.
        """
        # HOMEWORK: use a method from the policy to select an action
        action: int = self.policy.select_action(state)

        return action

    def learn(self, num_episodes: int = 10000) -> None:
        """
        Learns the optimal policy using Monte Carlo Exploring Starts.

        Args:
            num_episodes (int): The number of episodes to train for.
        """

        for episode_idx in range(num_episodes):

            # Print progress
            if episode_idx % 1000 == 0:
                print(f"Episode {episode_idx}/{num_episodes}")

            # HOMEWORK: Generate an episode.
            # There is a helper method for this within the superclass. Make sure to use exploring starts.
            episode: List[Tuple[Tuple[int, ...], int, float]] = self._generate_episode(exploring_starts=True)

            # Loop through the episode in reverse order, updating the q-values and policy
            returns: float = 0
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
                mc_error: float = returns - self.q_values.get(state, action)

                # HOMEWORK: Increment N(St, At).
                # self.state_action_stats tracks N(St, At), and has a method to update it.
                self.state_action_stats.update(state, action)

                # HOMEWORK: step_size = 1 / N(St, At)
                step_size: float = 1 / self.state_action_stats.get(state, action)

                # HOMEWORK: Calculate NewEstimate = OldEstimate + 1/N(St, At) * (Return - OldEstimate)
                new_value: float = self.q_values.get(state, action) + step_size * mc_error

                # Update action value array with new estimate
                self.q_values.update(state, action, new_value)

                # Update the policy
                self.policy.update(state, self.q_values)

            # Log the episode
            self.logger.log_episode()


def smooth(x: np.ndarray, window: int = 1000) -> np.ndarray:
    """
    Smooths a 1D array using a moving average with the specified window size.

    Args:
        x (np.ndarray): The input array to be smoothed.
        window (int): The size of the moving window.

    Returns:
        np.ndarray: The smoothed array.
    """
    return np.convolve(x, np.ones(window), 'valid') / window


def run(num_episodes: int = 50000) -> None:
    """
    Runs the Monte Carlo Exploring Starts agent on the Blackjack environment.
    """

    # Run parameters
    train_episodes: int = num_episodes
    # To generate more converged results similar to the lecture slides,
    # set num_episodes to 500000 when calling the run function.

    # Instantiate and learn the agent
    env: Env = gym.make("Blackjack-v1", sab=True)  # `sab` means rules following Sutton and Barto
    mc_control: MCExploringStartsAgent = MCExploringStartsAgent(env, gamma=1.0)
    mc_control.learn(num_episodes=train_episodes)

    # Plot the results
    plot_results(mc_control)

    # Plot total rewards
    plt.plot(smooth(mc_control.logger.total_rewards_per_episode))
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Monte Carlo Exploring Starts agent on the Blackjack environment."
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=50000,
        help="Number of episodes to train for. Use 500000 for more converged results."
    )
    args = parser.parse_args()

    run(num_episodes=args.num_episodes)
