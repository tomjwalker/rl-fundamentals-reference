"""
Simple implementation of the blocking maze gridworld on pp167 of Sutton and Barto (2018) for exploring Dyna-Q+ (
changing environments).

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
import numpy as np
from gymnasium.spaces import Discrete
# from gymnasium.spaces import Box

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import grid
matplotlib.use('TkAgg')


class BlockingMaze:

    def __init__(self):

        # "#" = standard, "S" = start, "G" = goal, "W" = wall

        self.initial_layout = np.array(
            [
                ["#", "#", "#", "#", "#", "#", "#", "#", "G"],
                ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
                ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
                ["W", "W", "W", "W", "W", "W", "W", "W", "#"],
                ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
                ["#", "#", "#", "S", "#", "#", "#", "#", "#"],
            ],
            dtype=object
        )

        self.layout = self.initial_layout

        self.action_space = Discrete(4)
        # self.observation_space = Box(low=0, high=self.layout.shape[1] - 1, shape=(2, ), dtype=int)
        self.observation_space = Discrete(self.layout.size)

        self.total_steps = 0

        self._state = None
        self.state = None
        self.episode_steps = None
        self.reset()

    def reset(self):
        """
        Resets the environment to its initial _state, and returns the initial observation.

         _state:
        [0, 0] || [0, 1] || [0, 2] || [0, 3] || [0, 4] || [0, 5] || [0, 6] || [0, 7] || [0, 8]
        [1, 0] || [1, 1] || [1, 2] || [1, 3] || [1, 4] || [1, 5] || [1, 6] || [1, 7] || [1, 8]
        [2, 0] || [2, 1] || [2, 2] || [2, 3] || [2, 4] || [2, 5] || [2, 6] || [2, 7] || [2, 8]
        [3, 0] || [3, 1] || [3, 2] || [3, 3] || [3, 4] || [3, 5] || [3, 6] || [3, 7] || [3, 8]
        [4, 0] || [4, 1] || [4, 2] || [4, 3] || [4, 4] || [4, 5] || [4, 6] || [4, 7] || [4, 8]
        [5, 0] || [5, 1] || [5, 2] || [5, 3] || [5, 4] || [5, 5] || [5, 6] || [5, 7] || [5, 8]

        state:
        0  || 1  || 2  || 3  || 4  || 5  || 6  || 7  || 8
        9  || 10 || 11 || 12 || 13 || 14 || 15 || 16 || 17
        18 || 19 || 20 || 21 || 22 || 23 || 24 || 25 || 26
        27 || 28 || 29 || 30 || 31 || 32 || 33 || 34 || 35
        36 || 37 || 38 || 39 || 40 || 41 || 42 || 43 || 44
        45 || 46 || 47 || 48 || 49 || 50 || 51 || 52 || 53
        """

        # Reset episode steps
        self.episode_steps = 0

        # Set the agent's initial position. 1st element is NumPy row (y-coordinate), 2nd element is NumPy column
        # (x-coordinate)
        row, col = np.where(self.layout == "S")
        start_index = [row[0], col[0]]
        self._state = np.array(start_index)

        self.state = self.flatten(self._state)

        # Return the initial observation
        return self.state, {}

    def flatten(self, state):
        """
        Flattens 2D _state (which indicates row and column) into a 1D state.
        """
        return state[0] * self.layout.shape[1] + state[1]

    def unflatten(self, state):
        """Unflattens 1D state into a 2D _state (which indicates row and column)."""
        return np.array([state // self.layout.shape[1], state % self.layout.shape[1]])

    def _change_layout(self):
        """
        Changes the layout of the environment.
        """

        if self.total_steps == 1000:

            self.layout = np.array(
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

    def step(self, action):
        """
        Takes an action, and returns a tuple (observation, reward, terminated, truncated=False, info).

        If the action is invalid, the agent remains in the same _state: given the action, if the next _state would be a
        wall, the agent remains in the same _state. Similarly, if the action would take the agent off the grid.
        """

        # Get the agent's current position
        agent_y, agent_x = self._state    # 1st element is NumPy row (y-coordinate), 2nd element is NumPy column

        # Layout changes after 1000 steps
        self._change_layout()

        # Get the next _state
        if action == 0:    # Move up
            _next_state = np.array([agent_y - 1, agent_x])
        elif action == 1:   # Move right
            _next_state = np.array([agent_y, agent_x + 1])
        elif action == 2:   # Move down
            _next_state = np.array([agent_y + 1, agent_x])
        elif action == 3:   # Move left
            _next_state = np.array([agent_y, agent_x - 1])
        else:
            raise ValueError(f"Invalid action {action}")

        # Check if the next _state is off the grid
        if _next_state[0] < 0 or _next_state[0] >= self.layout.shape[0] or _next_state[1] < 0 or _next_state[1] >= \
                self.layout.shape[1]:
            _next_state = self._state

        # Check if the next _state is a wall
        if self.layout[_next_state[0]][_next_state[1]] == "W":
            _next_state = self._state

        # Get the reward
        reward = 0
        if self.layout[_next_state[0]][_next_state[1]] == "G":
            reward = 1

        # Check if the episode is done
        terminated = self.layout[_next_state[0]][_next_state[1]] == "G"
        truncated = False
        # #
        # # # TODO: Remove this block (/replace with proper tests. Temp check)
        # next_state = self.flatten(_next_state)
        # if self.total_steps < 1000:
        #     if self.state == 44:    # State just below changing cell
        #         if action == 0:     # Move up
        #             assert next_state == 35    # Should be allowed to move up initially
        #         elif action == 1:   # Move right
        #             assert next_state == 44
        #         elif action == 2:   # Move down
        #             assert next_state == 53
        #         elif action == 3:   # Move left, off the grid
        #             assert next_state == 43
        #         else:
        #             raise ValueError(f"Invalid action {action} and next state {next_state} from start state 18")
        # if self.total_steps > 1000:
        #     if self.state == 44:    # Start state
        #         if action == 0:     # Move up
        #             assert next_state == 44    # **Should now be blocked by a wall**
        #         elif action == 1:   # Move right
        #             assert next_state == 44
        #         elif action == 2:   # Move down
        #             assert next_state == 53
        #         elif action == 3:   # Move left, off the grid
        #             assert next_state == 43
        #         else:
        #             raise ValueError(f"Invalid action {action} and next state {next_state} from start state 18")

        # Update the agent's position
        self._state = _next_state
        self.state = self.flatten(self._state)

        # Update total steps
        self.total_steps += 1

        # Return the next observation, reward, done flag, and info dict
        return self.state, reward, terminated, truncated, {}

    def render(self):
        """Visualizes the environment using Matplotlib."""

        # Create the plot
        fig, ax = plt.subplots()

        # Create a colormap for different cell types
        cmap = {
            "#": "black",
            "S": "orange",
            "G": "green",
            "W": "gray",
        }

        # Create a patch for each cell
        patches = []
        for i, row in enumerate(self.layout):
            for j, cell in enumerate(row):
                color = cmap[cell]
                patch = mpatches.Rectangle(xy=(j, i), width=1, height=1, color=color)
                patches.append(patch)

        # Place the agent's marker on top
        agent_y, agent_x = self._state
        agent_marker = mpatches.RegularPolygon(
            xy=(agent_x + 0.5, agent_y + 0.5), numVertices=5, color="red", radius=0.2
        )
        patches.append(agent_marker)

        # Set plot limits and labels

        ax.set_xticks(range(self.layout.shape[1] + 1))  # Tick at every column
        ax.set_yticks(range(self.layout.shape[1] + 1))  # Tick at every row
        ax.set_xlim(-0.5, self.layout.shape[1] + 0.5)
        ax.set_ylim(-0.5, self.layout.shape[0] + 0.5)

        grid(color="white", linestyle="-", linewidth=2)  # Add gridlines

        # Add all patches to the plot
        for patch in patches:
            ax.add_patch(patch)

        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")
        ax.set_title("BlockingMaze Environment")
        ax.invert_yaxis()

        # Display the plot
        plt.show()


def main():

    # Create the environment
    env = BlockingMaze()

    # Reset the environment
    env.reset()

    # Render the environment
    env.render()

    # # Take some random actions
    # num_actions = 10
    # for _ in range(num_actions):
    #
    #     # Take a random action
    #     action = env.action_space.sample()
    #     next_state, reward, terminated, truncated, _ = env.step(action)
    #
    #     # Render the environment
    #     env.render()
    #

    # Render the environment after the layout changes
    for _ in range(1001):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
    env.render()


if __name__ == "__main__":
    main()
