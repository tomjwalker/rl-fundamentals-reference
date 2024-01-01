"""
Simple implementation of the maze gridworld on pp165 of Sutton and Barto (2018).

Follows the Gymnasium API, so that it can be used with the same algorithms and tools: an environment class with the
following:
    - Attributes:
        - action_space: a gym.spaces.Discrete object, representing the action space
        - observation_space: a gym.spaces.Discrete object, representing the observation space
    - Methods:
        - reset(): resets the environment to its initial state, and returns the initial observation
        - step(action): takes an action, and returns a tuple (observation, reward, done, info)
        - render(): renders the environment
"""
import numpy as np
from gymnasium.spaces import Discrete

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import grid
matplotlib.use('TkAgg')


class Maze:

    def __init__(self):

        # "#" = standard, "S" = start, "G" = goal, "W" = wall
        layout = np.array(
            [
                ["#", "#", "#", "#", "#", "#", "#", "W", "G"],
                ["#", "#", "W", "#", "#", "#", "#", "W", "#"],
                ["S", "#", "W", "#", "#", "#", "#", "W", "#"],
                ["#", "#", "W", "#", "#", "#", "#", "#", "#"],
                ["#", "#", "#", "#", "#", "W", "#", "#", "#"],
                ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
            ],
            dtype=object
        )

        self.layout = layout

        self.action_space = Discrete(4)
        self.observation_space = Discrete(layout.size)

        self.state = None
        self.reset()

    def reset(self):
        """Resets the environment to its initial state, and returns the initial observation."""

        # Set the agent's initial position
        self.state = np.array([0, 2])

        # Return the initial observation
        return self.state

    def step(self, action):
        """
        Takes an action, and returns a tuple (observation, reward, done, info).

        If the action is invalid, the agent remains in the same state: given the action, if the next state would be a
        wall, the agent remains in the same state. Similarly if the action would take the agent off the grid.
        """

        # Get the agent's current position
        agent_x, agent_y = self.state

        # Get the agent's next position
        if action == 0:  # Up
            next_state = (agent_x, agent_y - 1)
        elif action == 1:  # Right
            next_state = (agent_x + 1, agent_y)
        elif action == 2:  # Down
            next_state = (agent_x, agent_y + 1)
        elif action == 3:  # Left
            next_state = (agent_x - 1, agent_y)
        else:
            raise ValueError(f"Invalid action {action}")

        # Check if the next state is a wall
        if self.layout[next_state] == "W":
            next_state = self.state

        # Check if the next state is off the grid
        if next_state[0] < 0 or next_state[0] >= self.layout.shape[1] or next_state[1] < 0 or next_state[1] >= \
                self.layout.shape[0]:
            next_state = self.state

        # Update the agent's position
        self.state = next_state

        # Get the reward
        reward = 0
        if self.layout[next_state] == "G":
            reward = 1

        # Check if the episode is done
        terminated = self.layout[next_state] == "G"
        truncated = False

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
        agent_x, agent_y = self.state
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
        ax.set_title("Maze Environment")
        ax.invert_yaxis()

        # Display the plot
        plt.show()


def main():

    # Create the environment
    env = Maze()

    # Reset the environment
    env.reset()

    # Render the environment
    env.render()

    # Take some random actions
    num_actions = 10
    for _ in range(num_actions):

        # Take a random action
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Render the environment
        env.render()


if __name__ == "__main__":
    main()
