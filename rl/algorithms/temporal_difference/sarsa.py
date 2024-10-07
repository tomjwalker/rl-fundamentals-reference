from rl.algorithms.common.td_agent import TemporalDifferenceAgent
from rl.common.results_logger import ResultsLogger


class Sarsa(TemporalDifferenceAgent):
    """
    SARSA (State-Action-Reward-State-Action) algorithm for Temporal Difference learning.

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
        self.name: str = "Sarsa"

    def learn(self, num_episodes: int = 500) -> None:
        """
        Trains the SARSA agent for a given number of episodes.

        Args:
            num_episodes (int): Number of episodes to train the agent.
        """
        for episode in range(num_episodes):

            # Initialise S
            state, _ = self.env.reset()

            # HOMEWORK: Choose A from S using policy derived from Q (epsilon-greedy)
            # N.B. Implement the act method in the TemporalDifferenceAgent superclass
            action: int = self.act(state)

            # Loop over each step of episode, until S is terminal
            done: bool = False
            while not done:

                # HOMEWORK: Make a step of the environment; observe R, S'
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # HOMEWORK: Choose A' from S' using policy derived from Q (epsilon-greedy)
                next_action: int = self.act(next_state)

                # HOMEWORK STARTS: (~3-4 lines).
                # Update Q(S, A), taking as target the TD target (R + gamma * Q(S', A'))
                # You ultimately want to update via self.q_values.update(...)
                td_target: float = reward + self.gamma * self.q_values.get(next_state, next_action)
                td_error: float = td_target - self.q_values.get(state, action)
                new_value: float = self.q_values.get(state, action) + self.alpha * td_error
                self.q_values.update(state, action, new_value)
                # HOMEWORK ENDS

                # HOMEWORK STARTS: S <- S', A <- A' (2 lines)
                state = next_state
                action = next_action
                # HOMEWORK ENDS

                # HOMEWORK: Add reward to episode reward log (self.logger.log_timestep(...))
                self.logger.log_timestep(reward)

                # HOMEWORK: If S is terminal, then episode is done (will exit "while" loop)
                done = terminated or truncated

            # Add episode reward to list
            self.logger.log_episode()
