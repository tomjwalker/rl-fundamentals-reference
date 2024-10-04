import numpy as np
from typing import Tuple, Union
from rl.common.q_value_table import QValueTable


class BasePolicy:
    def __init__(self, action_space: Union[int, Tuple[int]]) -> None:
        """
        Initialises the BasePolicy.

        Args:
            action_space (Union[int, Tuple[int]]): The number of possible actions.
        """
        self.action_space: Union[int, Tuple[int]] = action_space

    def select_action(self, state: Tuple[int, ...], q_values: QValueTable) -> int:
        """
        Abstract method to select an action based on the given state and Q-values.

        Args:
            state (Tuple[int, ...]): The current state of the environment.
            q_values (QValueTable): The Q-value table for the agent.

        Returns:
            int: The action to be taken.
        """
        raise NotImplementedError


class DeterministicPolicy:
    """
    A deterministic policy that selects the action with the highest value for each state.
    (π(s) → a = argmax_a Q(s, a))

    Args:
        state_shape (Tuple[int, ...]): The shape of the environment state space.
    """
    # TODO: unify with EpsilonGreedyPolicy / BasePolicy
    #   (including dtype)
    # TODO: "value" risks confusion with value function?
    def __init__(self, state_shape: Tuple[int, ...]) -> None:
        """
        Initialises the DeterministicPolicy.

        Args:
            state_shape (Tuple[int, ...]): The shape of the environment state space.
        """
        self.value: np.ndarray = np.zeros(state_shape, dtype=np.int8)

    def select_action(self, state: Tuple[int, ...]) -> int:
        """
        Selects an action based on the current policy.

        Args:
            state (Tuple[int, ...]): The current state of the environment.

        Returns:
            int: The action selected by the policy.
        """
        action: int = self.value[state]
        return action

    def update(
            self,
            state: Tuple[int, ...],
            q_values: QValueTable,
            ties: str = "random",
    ) -> None:
        """
        Updates the policy based on the Q-value table.

        Args:
            state (Tuple[int, ...]): The current state of the environment.
            q_values (QValueTable): The Q-value table for the agent.
            ties (str, optional): Strategy to break ties, default is "random".
        """
        self.value[state] = q_values.get_max_action(state, ties=ties)


class EpsilonGreedyPolicy(BasePolicy):
    """
    An epsilon-greedy policy that selects actions randomly with probability epsilon,
    and selects the action with the highest Q-value otherwise.

    Args:
        epsilon (float): The probability of selecting a random action.
        action_space (Union[int, Tuple[int]]): The number of possible actions.
    """
    def __init__(self, epsilon: float, action_space: Union[int, Tuple[int]]) -> None:
        """
        Initialises the EpsilonGreedyPolicy.

        Args:
            epsilon (float): The probability of selecting a random action.
            action_space (Union[int, Tuple[int]]): The number of possible actions.
        """
        super().__init__(action_space)
        self.epsilon: float = epsilon

    def select_action(
            self,
            state: Tuple[int, ...],
            q_values: QValueTable,
            ties: str = "random",
    ) -> int:
        """
        Selects an action based on epsilon-greedy strategy.

        Args:
            state (Tuple[int, ...]): The current state of the environment.
            q_values (QValueTable): The Q-value table for the agent.
            ties (str, optional): Strategy to break ties, default is "random".

        Returns:
            int: The action to be taken.
        """
        if np.random.random() < self.epsilon:
            action: int = np.random.choice(self.action_space)
            return action
        else:
            action: int = q_values.get_max_action(state, ties=ties)
            return action

    def compute_probs(self, state: Tuple[int, ...], q_values: QValueTable) -> np.ndarray:
        """
        Computes the action probabilities for the given state.

        Args:
            state (Tuple[int, ...]): The current state of the environment.
            q_values (QValueTable): The Q-value table for the agent.

        Returns:
            np.ndarray: The probability distribution over actions.
        """
        probs: np.ndarray = np.ones(self.action_space) * self.epsilon / self.action_space
        best_action: int = q_values.get_max_action(state)
        probs[best_action] = 1 - self.epsilon + self.epsilon / self.action_space
        return probs
