from rl.algorithms.common.base_agent import BaseAgent
from rl.common.policy import EpsilonGreedyPolicy
from rl.common.q_value_table import QValueTable
from rl.common.results_logger import ResultsLogger
from typing import Union


class TemporalDifferenceAgent(BaseAgent):
    """
    Superclass for all Temporal Difference agents. Contains common methods and attributes for all Temporal Difference agents.

    Args:
        env: The environment to interact with.
        gamma (float): Discount factor for future rewards.
        alpha (float): Learning rate.
        epsilon (float): Exploration parameter for epsilon-greedy policy.
        logger (ResultsLogger, optional): Logger for tracking results during training.
        random_seed (int, optional): Seed for reproducibility.
    """
    def __init__(
        self,
        env,
        gamma: float,
        alpha: float,
        epsilon: float,
        logger: Union[ResultsLogger, None] = None,
        random_seed: Union[int, None] = None
    ) -> None:
        super().__init__(env, gamma, random_seed)
        self.name: str = "Base Temporal Difference Agent"
        self.alpha: float = alpha
        self.epsilon: float = epsilon

        self.logger: ResultsLogger = logger if logger else ResultsLogger()

        # Initialise common Temporal Difference agent attributes
        self.q_values: Union[QValueTable, None] = None
        self.policy: Union[EpsilonGreedyPolicy, None] = None
        self.reset()

    def reset(self) -> None:
        """
        Resets the agent's attributes, including Q-values, policy, and logger.
        """
        self.q_values = QValueTable((self.env.observation_space.n,), self.env.action_space.n)
        self.policy = EpsilonGreedyPolicy(self.epsilon, self.env.action_space.n)
        self.logger.reset()

    def act(self, state: int) -> int:
        """
        Selects an action from the policy based on the given state.

        Args:
            state (int): The current state of the environment.

        Returns:
            int: The action selected by the policy.
        """
        # HOMEWORK: Select the action from the policy
        action: int = self.policy.select_action(state, self.q_values)
        return action

    def learn(self, num_episodes: int = 500) -> None:
        """
        Abstract method for training the agent. Should be implemented by subclasses.

        Args:
            num_episodes (int): Number of episodes to train the agent.
        """
        raise NotImplementedError
