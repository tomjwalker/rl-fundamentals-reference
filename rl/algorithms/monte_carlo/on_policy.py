"""
"""
# TODO: Figure out strange policy plots
# TODO: Look into this: https://trevormcguire.medium.com/blackjack-stocks-and-reinforcement-learning-ea4014115aeb


from rl.algorithms.monte_carlo.viz import plot_results
from rl.algorithms.common.mc_agent import MonteCarloAgent
from rl.common.policy import EpsilonGreedyPolicy

import gymnasium as gym
from typing import Union, Tuple
from gymnasium import Env
from rl.common.results_logger import ResultsLogger

import matplotlib
matplotlib.use('TkAgg')


class MCOnPolicy(MonteCarloAgent):
    """
    Monte Carlo On-Policy control implementation.
    Uses an epsilon-greedy policy to explore and learn an optimal policy for the given environment.

    Args:
        env (Union[Env, object]): The environment to interact with.
        gamma (float): Discount factor for future rewards.
        epsilon (float, optional): Exploration parameter for epsilon-greedy policy.
        logger (ResultsLogger, optional): Logger for tracking results during training.
        random_seed (int, optional): Seed for reproducibility.
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
        Initialises the MCOnPolicy agent.

        Args:
            env (Union[Env, object]): The environment to interact with.
            gamma (float): Discount factor for future rewards.
            epsilon (float, optional): Exploration parameter for epsilon-greedy policy.
            logger (ResultsLogger, optional): Logger for tracking results during training.
            random_seed (int, optional): Seed for reproducibility.
        """
        super().__init__(env, gamma, epsilon, logger, random_seed)

        self.name: str = "MC On-Policy"  # For plotting
        self.policy: Union[EpsilonGreedyPolicy, None] = None
        self.reset()

    def reset(self) -> None:
        """
        Resets the agent's attributes, including the policy and q-values.
        Initialises an epsilon-greedy policy for on-policy Monte Carlo control.
        """
        super().reset()

        # Policy method specific to On-Policy and Off-Policy MC (not ES)
        self.policy = EpsilonGreedyPolicy(self.epsilon, self.env.action_space.n)

    def act(self, state: Tuple[int, ...]) -> int:
        """
        Selects an action based on the epsilon-greedy policy.

        Args:
            state (Tuple[int, ...]): The current state of the environment.

        Returns:
            int: The action selected by the policy.
        """
        # HOMEWORK: sample an action from the epsilon-greedy policy
        action: int = self.policy.select_action(state, self.q_values)

        return action

    def learn(self, num_episodes: int = 10000) -> None:
        """
        Learns an optimal policy using Monte Carlo on-policy control.

        Args:
            num_episodes (int): The number of episodes to train the agent for.
        """
        for episode_idx in range(num_episodes):
            if episode_idx % 1000 == 0:
                print(f"Episode {episode_idx}/{num_episodes}")

            # Generate an episode
            # HOMEWORK: Generate an episode.
            # In the Exploring Starts module, you used `exploring_starts=True`.
            # Here, we want to use `exploring_starts=False`.
            episode = self._generate_episode(exploring_starts=False)

            # Loop through the episode in reverse order, updating the q-values and policy
            # HOMEWORK STARTS: (~10 lines of code) Update the q-values by looping through the episode in reverse order.
            # This part of the code is almost identical to the Exploring Starts module learn method,
            # but note that we do not need the final line to update the policy here.
            returns: float = 0
            for t, (state, action, reward) in enumerate(reversed(episode)):
                returns = self.gamma * returns + reward

                # If the S_t, A_t pair has been seen before, continue.
                if self._is_subelement_present((state, action), episode[:len(episode) - t - 1]):
                    continue

                # Update the q-value for this state-action pair
                # NewEstimate <- OldEstimate + 1/N(St, At) * (Return - OldEstimate)
                mc_error: float = returns - self.q_values.get(state, action)
                self.state_action_stats.update(state, action)  # Get N(St, At)
                step_size: float = 1 / self.state_action_stats.get(state, action)
                new_value: float = self.q_values.get(state, action) + step_size * mc_error
                self.q_values.update(state, action, new_value)
            # HOMEWORK ENDS

            # Log the episode
            self.logger.log_episode()


def run(num_episodes: int = 50000) -> None:
    """
    Runs the MCOnPolicy agent on the Blackjack environment and plots the results.
    """
    # Run parameters
    train_episodes: int = num_episodes

    # Create the environment
    env: Env = gym.make("Blackjack-v1", sab=True)  # `sab` means rules following Sutton and Barto
    mc_control: MCOnPolicy = MCOnPolicy(env, epsilon=0.1, gamma=1.0)
    mc_control.learn(num_episodes=train_episodes)

    # Plot the results
    plot_results(mc_control)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the MCOnPolicy agent on the Blackjack environment.")
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=50000,
        help="Number of episodes to train for. Use a larger number for more convergence."
    )
    args = parser.parse_args()

    run(num_episodes=args.num_episodes)
