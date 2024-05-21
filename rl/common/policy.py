import numpy as np
from typing import Tuple, Union
from rl.common.q_value_table import QValueTable


class EpsilonGreedyPolicy:
    def __init__(self, epsilon: float, action_space: Union[int, Tuple[int]]) -> None:
        self.epsilon = epsilon
        self.action_space = action_space

    def select_action(self, state: int, q_values: QValueTable) -> int:
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return q_values.get_max_action(state)

    def compute_probs(self, state: int, q_values: QValueTable) -> np.ndarray:
        probs = np.ones(self.action_space) * self.epsilon / self.action_space
        best_action = q_values.get_max_action(state)
        probs[best_action] = 1 - self.epsilon + self.epsilon / self.action_space
        return probs
