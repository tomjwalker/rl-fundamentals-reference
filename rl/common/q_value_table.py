import numpy as np
from rl.utils.general import argmax


class QValueTable:
    def __init__(self, state_space_size, action_space_size):
        self.q_values = np.zeros((state_space_size, action_space_size))

    def get(self, state, action=None):
        return self.q_values[state, action] if action is not None else self.q_values[state, :]

    def update(self, state, action, value):
        self.q_values[state, action] = value

    def get_max_action(self, state):
        return argmax(self.q_values[state, :])
