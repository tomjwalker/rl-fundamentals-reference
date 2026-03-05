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


class DeterministicPolicy(BasePolicy):
    """
    A deterministic policy that selects the action with the highest value for each state.

    Args:
        state_shape (Tuple[int, ...]): The shape of the environment state space.
        dtype (type, optional): The data type for the action map. Default is np.int8.
    """

    def __init__(self, state_shape: Tuple[int, ...], dtype: type = np.int8) -> None:
        """
        Initialises the DeterministicPolicy.

        Args:
            state_shape (Tuple[int, ...]): The shape of the environment state space.
            dtype (type, optional): The data type for the action map. Default is np.int8.
        """
        super().__init__(action_space=int(np.prod(state_shape)))
        self.action_map: np.ndarray = np.zeros(state_shape, dtype=dtype)

    def select_action(self, state: Tuple[int, ...]) -> int:
        """
        Selects an action based on the current policy.

        Args:
            state (Tuple[int, ...]): The current state of the environment.

        Returns:
            int: The action selected by the policy.
        """
        action: int = self.action_map[state]
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
        self.action_map[state] = q_values.get_max_action(state, ties=ties)


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

        action = q_values.get_max_action(state, ties=ties)
        return action

    def compute_probs(self, state: Tuple[int, ...], q_values: QValueTable) -> np.ndarray:
        """
        Computes the action probabilities for the given state.

        As this is epsilon-greedy, this means the probabilities are:
            - epsilon / action_space for all actions
            - the greedy mass (1 - epsilon) shared uniformly across tied best actions

        Args:
            state (Tuple[int, ...]): The current state of the environment.
            q_values (QValueTable): The Q-value table for the agent.

        Returns:
            np.ndarray: The probability distribution over actions.
        """
        probs: np.ndarray = np.ones(self.action_space) * self.epsilon / self.action_space
        action_values: np.ndarray = q_values.get(state)
        max_value: float = np.max(action_values)
        greedy_actions: np.ndarray = np.flatnonzero(action_values == max_value)

        # Split the greedy mass uniformly across all tied best actions.
        probs[greedy_actions] += (1 - self.epsilon) / len(greedy_actions)
        return probs
