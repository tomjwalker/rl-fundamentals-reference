from rl.algorithms.common.td_agent import TemporalDifferenceAgent
import numpy as np
from rl.common.results_logger import ResultsLogger
from typing import Union


class ExpectedSarsa(TemporalDifferenceAgent):
    """
    Expected SARSA algorithm for Temporal Difference learning.

    Args:
        env: The environment to interact with.
        alpha (float): Learning rate.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Exploration parameter for epsilon-greedy policy.
        logger (ResultsLogger, optional): Logger for tracking results during training.
        random_seed (int, optional): Seed for reproducibility.
    """

    def __init__(
        self,
        env,
        alpha: float = 0.5,
        gamma: float = 1.0,
        epsilon: float = 0.1,
        logger: Union[ResultsLogger, None] = None,
        random_seed: Union[int, None] = None
    ) -> None:
        super().__init__(env, gamma, alpha, epsilon, logger, random_seed)
        self.name: str = "Expected Sarsa"

    def learn(self, num_episodes: int = 500) -> None:
        """
        Trains the Expected SARSA agent for a given number of episodes.

        Args:
            num_episodes (int): Number of episodes to train the agent.
        """
        # HOMEWORK BEGINS: Implement the Expected Sarsa algorithm (~14 lines). Refer to the Sarsa implementation.

        for episode in range(num_episodes):

            # Initialise S
            state, _ = self.env.reset()

            # Loop over each step of episode, until S is terminal
            done: bool = False
            while not done:

                # Choose A from S using policy derived from Q (epsilon-greedy)
                action: int = self.act(state)

                # Take action A, observe R, S'
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Compute expected value of Q(S', A') under policy pi: sum_a pi(a|S') * Q(S', a)
                action_values: np.ndarray = self.q_values.get(next_state)
                prob_values: np.ndarray = self.policy.compute_probs(next_state, self.q_values)
                expected_value: float = np.sum(prob_values * action_values)

                # Update Q(S, A), taking as target the expected-sarsa TD target (R + gamma * sum_a pi(a|S') * Q(S', a))
                td_target: float = reward + self.gamma * expected_value
                td_error: float = td_target - self.q_values.get(state, action)
                new_value: float = self.q_values.get(state, action) + self.alpha * td_error
                self.q_values.update(state, action, new_value)

                # S <- S', A <- A'
                state = next_state

                # Add reward to episode reward
                self.logger.log_timestep(reward)

                # If S is terminal, then episode is done (exit loop)
                done = terminated or truncated

            # Add episode reward to list
            self.logger.log_episode()

        # HOMEWORK ENDS
