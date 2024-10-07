# TODO: plot of policies half way through second episode

from rl.environment.planning.planning_maze import Maze
import pandas as pd

# Import algorithms
from rl.algorithms.planning.dyna import Dyna

import matplotlib
from matplotlib import pyplot as plt
import argparse

matplotlib.use('TkAgg')

def run(train_episodes: int, gamma: float, alpha: float, epsilon: float, planning_steps: list) -> None:
    """
    Run the Dyna experiment in the Maze environment with different planning step configurations.

    Args:
        train_episodes (int): Number of episodes to train the agent.
        gamma (float): Discount factor for the agent.
        alpha (float): Learning rate for the agent.
        epsilon (float): Probability of choosing a random action (epsilon-greedy policy).
        planning_steps (list): List of planning steps to run (e.g., [0, 5, 50]).
    """
    # Run parameters
    run_specs = {
        "planning steps": planning_steps,
        "colour": ["blue", "green", "red"],
        "label": [f"{steps} planning steps" for steps in planning_steps],
    }
    run_specs = pd.DataFrame(run_specs)

    for i, row in run_specs.iterrows():
        # Create the environment
        env = Maze()

        # Create and learn the agent
        rl_loop = Dyna(env, gamma=gamma, alpha=alpha, epsilon=epsilon, n_planning_steps=row["planning steps"],
                       random_seed=42)
        rl_loop.learn(num_episodes=train_episodes)

        # Plot the results
        plt.plot(rl_loop.logger.steps_per_episode, color=row["colour"], label=row["label"])

    plt.ylim(bottom=0, top=800)    # Set y-limits after all plots are generated
    plt.xlabel("Episode")
    plt.ylabel("Episode steps")
    plt.title(f"Episode steps for Dyna agent (gamma={gamma})")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dyna agent in the Maze environment and plot the results.")
    parser.add_argument(
        '--train_episodes',
        type=int,
        default=50,
        help="Number of episodes to train the agent."
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.95,
        help="Discount factor for the agent."
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.1,
        help="Learning rate for the agent."
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.1,
        help="Probability of choosing a random action (epsilon-greedy policy)."
    )
    parser.add_argument(
        '--planning_steps',
        type=int,
        nargs='+',
        default=[0, 5, 50],
        help="List of planning steps to run (e.g., 0 5 50)."
    )
    args = parser.parse_args()

    run(
        train_episodes=args.train_episodes,
        gamma=args.gamma,
        alpha=args.alpha,
        epsilon=args.epsilon,
        planning_steps=args.planning_steps
    )
