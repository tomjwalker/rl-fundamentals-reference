"""
Simple implementation of the shortcut maze gridworld on pp167 of Sutton and Barto (2018) for exploring Dyna-Q+ (
changing environments).

Only change is when and how gridworld transitions to new layout.

Follows the Gymnasium API, so that it can be used with the same algorithms and tools: an environment class with the
following:
    - Attributes:
        - action_space: a gym.spaces.Discrete object, representing the action space
        - observation_space: a gym.spaces.Discrete object, representing the observation space
    - Methods:
        - reset(): resets the environment to its initial _state, and returns the initial observation
        - step(action): takes an action, and returns a tuple (observation, reward, done, info)
        - render(): renders the environment
"""
from rl.environment.planning.blocking_maze import BlockingMaze

import numpy as np
from gymnasium.spaces import Discrete

import matplotlib

matplotlib.use('TkAgg')


class ShortcutMaze(BlockingMaze):

    def __init__(self):

        super().__init__()

        self.name = "Shortcut Maze"

        self.initial_layout = np.array(
            [
                ["#", "#", "#", "#", "#", "#", "#", "#", "G"],
                ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
                ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
                ["#", "W", "W", "W", "W", "W", "W", "W", "W"],
                ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
                ["#", "#", "#", "S", "#", "#", "#", "#", "#"],
            ],
            dtype=object
        )

        self.layout = self.initial_layout

        self.action_space = Discrete(4)
        self.observation_space = Discrete(self.layout.size)

        self._state = None
        self.state = None
        self.reset()

    def _change_layout(self):
        """
        Changes the layout of the environment.
        """
        if self.total_steps == 3000:
            self.layout = np.array(
                [
                    ["#", "#", "#", "#", "#", "#", "#", "#", "G"],
                    ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
                    ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
                    ["#", "W", "W", "W", "W", "W", "W", "W", "#"],
                    ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
                    ["#", "#", "#", "S", "#", "#", "#", "#", "#"],
                ],
                dtype=object
            )


if __name__ == "__main__":

    env = ShortcutMaze()

    # Render initial layout
    env.render()

    # Render layout after 3000 steps
    for _ in range(3001):
        env.step(env.action_space.sample())
    env.render()
