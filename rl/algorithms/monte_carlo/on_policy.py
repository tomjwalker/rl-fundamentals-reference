"""
TODO:
    - Figure out strange policy plots
    - Look into this: https://trevormcguire.medium.com/blackjack-stocks-and-reinforcement-learning-ea4014115aeb
"""


from rl.algorithms.monte_carlo.viz import plot_results
from rl.algorithms.common.mc_agent import MonteCarloAgent
from rl.common.policy import EpsilonGreedyPolicy

import gymnasium as gym

from typing import Union
from gymnasium import Env
from rl.common.results_logger import ResultsLogger

import matplotlib
matplotlib.use('TkAgg')


class MCOnPolicy(MonteCarloAgent):

    def __init__(
            self,
            env: Union[Env, object],
            gamma: float,
            epsilon: float = None,
            logger: ResultsLogger = None,
            random_seed: int = None,
    ):
        super().__init__(env, gamma, epsilon, logger, random_seed)

        self.name = "MC On-Policy"    # For plotting

        self.policy = None
        self.reset()

    def reset(self):

        super().reset()

        # Policy method specific to On-Policy and Off-Policy MC (not ES)
        self.policy = EpsilonGreedyPolicy(self.epsilon, self.env.action_space.n)

    def act(self, state):
        return self.policy.select_action(state, self.q_values)

    def learn(self, num_episodes=10000):

        for episode_idx in range(num_episodes):
            if episode_idx % 1000 == 0:
                print(f"Episode {episode_idx}/{num_episodes}")

            # Generate an episode
            episode = self._generate_episode(exploring_starts=False)

            # Loop through the episode in reverse order, updating the q-values and policy
            g = 0
            for t, (state, action, reward) in enumerate(reversed(episode)):
                g = self.gamma * g + reward

                # If the S_t, A_t pair has been seen before, continue.
                if self._is_subelement_present((state, action), episode[:len(episode) - t - 1]):
                    continue

                # Update the q-value for this state-action pair
                # NewEstimate <- OldEstimate + 1/N(St, At) * (Return - OldEstimate)
                mc_error = g - self.q_values.get(state, action)
                self.state_action_counts.update(state, action)    # Get N(St, At)
                step_size = 1 / self.state_action_counts.get(state, action)
                new_value = self.q_values.get(state, action) + step_size * mc_error
                self.q_values.update(state, action, new_value)

        # Log the episode
        self.logger.log_episode()


def run():

    # Run parameters
    train_episodes = 50000

    # Create the environment
    env = gym.make("Blackjack-v1", sab=True)  # `sab` means rules following Sutton and Barto
    mc_control = MCOnPolicy(env, epsilon=0.1, gamma=1.0)
    mc_control.learn(num_episodes=train_episodes)

    # Plot the results
    plot_results(mc_control)


if __name__ == "__main__":
    run()
