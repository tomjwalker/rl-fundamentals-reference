# TODO: plot of policies half way through second episode


from rl.environment.planning.planning_maze import Maze
import pandas as pd

# Import algorithms
from rl.algorithms.planning.dyna import Dyna

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')


def run():

    # Run parameters
    train_episodes = 50
    gamma = 0.95
    alpha = 0.1
    epsilon = 0.1
    run_specs = {
        "planning steps": [0, 5, 50],
        "colour": ["blue", "green", "red"],
        "label": ["0 planning steps (direct RL)", "5 planning steps", "50 planning steps"],
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
    run()
