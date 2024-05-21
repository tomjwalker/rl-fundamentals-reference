import matplotlib
from matplotlib import pyplot as plt
from rl.algorithms.common.td_agent import TemporalDifferenceAgent
from typing import Tuple
matplotlib.use('TkAgg')


def visualise_q(agent: TemporalDifferenceAgent, grid_shape: Tuple[int, int] = (4, 12)) -> None:
    """
    Visualises the Q-values: reshapes to the cliff-walking environment shape, and plots:
    - grid of states (each cell represents a state), coloured by the Q-value of each action
    - arrow pointing in the direction of the action with the highest Q-value for each state (in the centre of the cell)
    """
    # Reshape q_values to grid_shape: new dimensions are (grid_shape[0], grid_shape[1], n_actions)
    q_values_grid = agent.q_values.q_values.reshape(grid_shape[0], grid_shape[1], agent.env.action_space.n)
    state_values = q_values_grid.max(axis=2)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot V-values
    plt.imshow(state_values, cmap='viridis')

    # Add arrows
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            # Get best action for state (i, j)
            best_action = q_values_grid[i, j].argmax()
            # Add arrow
            if best_action == 0:
                plt.arrow(j, i, 0, -0.4, head_width=0.1, head_length=0.1, fc='k', ec='k')
            elif best_action == 1:
                plt.arrow(j, i, 0, 0.4, head_width=0.1, head_length=0.1, fc='k', ec='k')
            elif best_action == 2:
                plt.arrow(j, i, -0.4, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
            elif best_action == 3:
                plt.arrow(j, i, 0.4, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')

    # Set up plot
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Q-values and policy")

    # Show plot
    plt.show()


# Debug
if __name__ == "__main__":
    import numpy as np
    import gymnasium as gym

    env = gym.make("CliffWalking-v0")
    td_agent = TemporalDifferenceAgent(env, gamma=1.0, alpha=0.5, epsilon=0.1)
    # Replace agent's q_values with random values
    td_agent.q_values.q_values = np.random.rand(env.observation_space.n, env.action_space.n)
    visualise_q(td_agent)


