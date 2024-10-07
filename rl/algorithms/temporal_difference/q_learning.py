from rl.algorithms.common.td_agent import TemporalDifferenceAgent
from rl.common.results_logger import ResultsLogger


class QLearning(TemporalDifferenceAgent):
    """
    Q-Learning algorithm for Temporal Difference learning.

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
        logger: ResultsLogger = None,
        random_seed: int = None
    ) -> None:
        super().__init__(env, gamma, alpha, epsilon, logger, random_seed)
        self.name: str = "Q-Learning"

    def learn(self, num_episodes: int = 500) -> None:
        """
        Trains the Q-Learning agent for a given number of episodes.

        Args:
            num_episodes (int): Number of episodes to train the agent.
        """
        # HOMEWORK BEGINS: Implement the Q-learning algorithm (~14 lines). Refer to the Sarsa implementation.

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

                # Update Q(S, A), taking as target the q-learning TD target (R + gamma * max_a Q(S', a))
                td_target: float = reward + self.gamma * self.q_values.get(next_state, self.q_values.get_max_action(next_state))
                td_error: float = td_target - self.q_values.get(state, action)
                new_value: float = self.q_values.get(state, action) + self.alpha * td_error
                self.q_values.update(state, action, new_value)

                # S <- S'
                state = next_state

                # Add reward to episode reward
                self.logger.log_timestep(reward)

                # If S is terminal, then episode is done (exit loop)
                done = terminated or truncated

            # Add episode reward to list
            self.logger.log_episode()

        # HOMEWORK ENDS
