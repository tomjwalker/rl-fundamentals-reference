import numpy as np
from rl.utils.general import argmax_ties_random, argmax_ties_last
from typing import Tuple, Union


# TODO: make even more general with vector actions

class QValueTable:
    def __init__(self, state_space_shape: Union[int, Tuple[int, ...]], action_space_shape: int):
        self.values = np.zeros(state_space_shape + (action_space_shape,))

    def get(self, state: Union[int, Tuple[int, ...]], action: Union[int, None] = None) -> Union[float, np.ndarray]:
        if action is not None:
            # If action is specified, return the value of that action in the state
            return self.values[state][action]
        else:
            # If action is not specified, return the values of all actions in the state
            return self.values[state]

    def update(self, state: Union[int, Tuple[int, ...]], action: int, value: float) -> None:
        self.values[state][action] = value

    def get_max_action(self, state: Union[int, Tuple[int, ...]], ties: str = "random") -> int:
        if ties == "random":
            return argmax_ties_random(self.values[state])
        elif ties == "last":
            return argmax_ties_last(self.values[state])
        elif ties == "first":
            return np.argmax(self.values[state])
        else:
            raise ValueError(f"Unknown tie-breaking method: {ties}")
