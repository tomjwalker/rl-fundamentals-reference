"""
"""
# TODO: figure out strange policy plots


from rl.algorithms.monte_carlo.viz import plot_results
from rl.algorithms.common.mc_agent import MonteCarloAgent
from rl.common.policy import EpsilonGreedyPolicy, DeterministicPolicy

import gymnasium as gym

from typing import Union, Tuple, List
from gymnasium import Env
from rl.common.results_logger import ResultsLogger

import matplotlib
matplotlib.use('TkAgg')


class MCOffPolicy(MonteCarloAgent):
    """
    Monte Carlo Off-Policy control implementation.
    Uses an epsilon-greedy behaviour policy and a deterministic target policy to explore and learn an optimal policy for
    the given environment.

    Args:
        env (Union[Env, object]): The environment to interact with.
        gamma (float): Discount factor for future rewards.
        epsilon (float, optional): Exploration parameter for epsilon-greedy behaviour policy.
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
        super().__init__(env, gamma, epsilon, logger, random_seed)

        self.name: str = "MC Off-Policy"
        self.policy: Union[DeterministicPolicy, None] = None
        self.behaviour_policy: Union[EpsilonGreedyPolicy, None] = None
        self.reset()

    def reset(self) -> None:
        """
        Resets the agent's attributes, including the target and behaviour policies.
        Initialises a deterministic target policy and an epsilon-greedy behaviour policy for
        off-policy Monte Carlo control.
        """
        super().reset()
        self.policy = DeterministicPolicy(self.state_shape)
        self.behaviour_policy = EpsilonGreedyPolicy(self.epsilon, self.env.action_space.n)

    def act(self, state: Tuple[int, ...]) -> int:
        """
        Selects an action based on the behaviour policy.
        """
        return self.behaviour_policy.select_action(state, self.q_values)

    def _update_q_and_pi(self, episode: List[Tuple[Tuple[int, ...], int, float]]) -> None:
        """
        Updates q-values using weighted importance sampling and updates the target policy.
        """
        returns: float = 0
        weights: float = 1

        for state, action, reward in reversed(episode):
            returns = self.gamma * returns + reward
            self.state_action_stats.update_importance_sampling(state, action, weights)

            step_size: float = weights / self.state_action_stats.get(state, action)
            new_value: float = self.q_values.get(state, action) + step_size * (returns - self.q_values.get(state, action))
            self.q_values.update(state, action, new_value)
            self.policy.update(state, self.q_values, ties="last")

            if action != self.policy.select_action(state):
                break

            weights *= 1 / self.behaviour_policy.compute_probs(state, self.q_values)[action]

    def learn(self, num_episodes: int = 10000) -> None:
        """
        Learns an optimal policy using Monte Carlo off-policy control.

        Args:
            num_episodes (int): The number of episodes to train the agent for.
        """
        for episode_idx in range(num_episodes):
            if episode_idx % 1000 == 0:
                print(f"Episode {episode_idx}/{num_episodes}")

            episode = self._generate_episode(exploring_starts=False)
            self._update_q_and_pi(episode)
            self.logger.log_episode()


def run(num_episodes: int = 50000) -> None:
    """
    Runs the MCOffPolicy agent on the Blackjack environment and plots the results.

    Args:
        num_episodes (int): The number of episodes to train the agent for.
    """
    env: Env = gym.make("Blackjack-v1", sab=True)
    rl_loop: MCOffPolicy = MCOffPolicy(env, epsilon=0.1, gamma=1.0)
    rl_loop.learn(num_episodes=num_episodes)
    plot_results(rl_loop)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the MCOffPolicy agent on the Blackjack environment.")
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=50000,
        help="Number of episodes to train for. Use a larger number for more convergence."
    )
    args = parser.parse_args()

    run(num_episodes=args.num_episodes)
