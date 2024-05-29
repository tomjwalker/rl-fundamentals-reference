from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import random
from gymnasium import Env


# TODO: check type hints after refactoring


class BaseAgent(ABC):
    """
    Base class from which all agents inherit. Contains common methods and attributes for all the following agents:
    - Monte Carlo // On-policy
    - Monte Carlo // Off-policy
    - Temporal difference // SARSA
    - Temporal difference // Q-learning
    - Temporal difference // Expected SARSA
    - Planning // Dyna
    - Planning // Dyna-Q+
    """
    def __init__(self, env: Union[Env, object], gamma: float, random_seed: Union[None, int] = None) -> None:

        self.name = "Base Agent"
        self.env = env
        self.gamma = gamma
        self.random_seed = random_seed

        if random_seed is not None:
            self._set_random_seed()

        # TODO: retain this?
        self.policy = None

    def _set_random_seed(self):
        """
        Sets the random seed for reproducibility.
        """
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    @abstractmethod
    def reset(self):
        """
        Separating out a reset method from the __init__ method allows for the agent to be reset without having to be
        re-instantiated. Can help for reusing the same agent across multiple experiments, clarity, and efficiency.
        """
        raise NotImplementedError

    @abstractmethod
    def act(self, state: int) -> int:
        """Choose an action based on the current state and policy."""
        raise NotImplementedError

    @abstractmethod
    def learn(self, num_episodes: int = 500) -> None:
        """Train the agent over a specified number of episodes."""
        raise NotImplementedError
