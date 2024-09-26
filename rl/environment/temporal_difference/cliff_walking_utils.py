import matplotlib
from matplotlib import pyplot as plt
from rl.algorithms.common.td_agent import TemporalDifferenceAgent
# from rl.common.results_logger import ResultsLogger
import numpy as np
from typing import Tuple
matplotlib.use('TkAgg')


def visualise_q(agent: TemporalDifferenceAgent, grid_shape: Tuple[int, int] = (4, 12), ax: plt.Axes = None) -> None:
    """
    Visualises the Q-values: reshapes to the cliff-walking environment shape, and plots:
    - grid of states (each cell represents a state), coloured by the Q-value of each action
    - arrow pointing in the direction of the action with the highest Q-value for each state (in the centre of the cell)
    """
    # Reshape values to grid_shape: new dimensions are (grid_shape[0], grid_shape[1], n_actions)
    q_values_grid = agent.q_values.values.reshape(grid_shape[0], grid_shape[1], agent.env.action_space.n)
    state_values = q_values_grid.max(axis=2)

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # Plot V-values
    ax.imshow(state_values, cmap='viridis')

    # Add arrows
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            # Get best action for state (i, j)
            best_action = q_values_grid[i, j].argmax()
            if state_values[i, j] == 0:
                continue    # Skip terminal states
            # Add arrow
            if best_action == 0:    # Up
                ax.arrow(j, i, 0, -0.4, head_width=0.1, head_length=0.1, fc='k', ec='k')
            elif best_action == 1:    # Right
                ax.arrow(j, i, 0.4, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
            elif best_action == 2:   # Down
                ax.arrow(j, i, 0, 0.4, head_width=0.1, head_length=0.1, fc='k', ec='k')
            elif best_action == 3:  # Left
                ax.arrow(j, i, -0.4, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')

    # Mask terminal states (cliff and goal) - cover cell with grey
    mask_all_zero_actions = state_values == 0
    cmap_mask = plt.cm.colors.ListedColormap(["none", "gray"])
    ax.imshow(mask_all_zero_actions, cmap=cmap_mask)

    # Set up plot
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Q-values for {agent.name} agent")

    # Show plot if no axis is provided
    if ax is None:
        plt.show()


def visualise_state_visits(state_visits: np.ndarray, grid_shape: Tuple[int, int] = (4, 12),
                           ax: plt.Axes = None) -> None:
    """
    Visualises the state visitation count for the cliff-walking environment.

    :param state_visits: 1D array of state visitation counts
    :param grid_shape: Shape of the cliff-walking grid
    :param ax: Matplotlib axes to plot on. If None, a new figure is created.
    """
    # Reshape visits to grid_shape
    visits_grid = state_visits.reshape(grid_shape)

    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # Plot visitation counts
    im = ax.imshow(visits_grid, cmap='viridis')

    # Add colorbar
    plt.colorbar(im, ax=ax, label='Visit count')

    # Add visit counts as text in each cell
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            ax.text(j, i, str(visits_grid[i, j]),
                    ha='center', va='center', color='w', fontweight='bold')

    # Mask terminal states (cliff and goal) - cover cell with grey
    mask_terminal = visits_grid == 0
    cmap_mask = plt.cm.colors.ListedColormap(["none", "gray"])
    ax.imshow(mask_terminal, cmap=cmap_mask)

    # Set up plot
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("State visitation count")

    # Show plot if no axis is provided
    if ax is None:
        plt.show()


# Debug
if __name__ == "__main__":
    import numpy as np
    import gymnasium as gym

    env = gym.make("CliffWalking-v0")
    td_agent = TemporalDifferenceAgent(env, gamma=1.0, alpha=0.5, epsilon=0.1)
    # Replace agent's values with random values
    td_agent.q_values.values = np.random.rand(env.observation_space.n, env.action_space.n)
    visualise_q(td_agent)
    plt.show()
