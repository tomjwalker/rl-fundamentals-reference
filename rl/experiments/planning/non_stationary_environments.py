# Import relevant gridworld environment (local implementation of blocking and shortcut maze gridworlds, pp167 of Sutton
# and Barto (2018))
from rl.environment.planning.blocking_maze import BlockingMaze
from rl.environment.planning.shortcut_maze import ShortcutMaze

# Import algorithms
from rl.algorithms.planning.dyna import Dyna
from rl.algorithms.planning.dyna_plus import DynaPlus

from rl.simulation.trial import Trial

import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')


def run():

    # Run parameters
    env_specs = {
        "BlockingMaze": {
            "env": BlockingMaze,
            "train_episodes": 400,
            "ylim": 150,
            "xlim": 3000,
        },
        "ShortcutMaze": {
            "env": ShortcutMaze,
            "train_episodes": 400,
            "ylim": 400,
            "xlim": 6000,
        },
    }

    environment_name = "ShortcutMaze"

    # TODO:
    n_runs = 30

    # train_episodes = 3000
    train_episodes = env_specs[environment_name]["train_episodes"]
    gamma = 0.95
    epsilon = 0.1
    alpha = 0.5
    planning_steps = 50
    run_specs = {
        "model": [Dyna, DynaPlus],
        "colour": ["blue", "red"],
        "label": ["Dyna-Q", "Dyna-Q+"],
    }
    run_specs = pd.DataFrame(run_specs)

    for i, row in run_specs.iterrows():

        # Create the environment
        env = env_specs[environment_name]["env"]()

        # Create and learn the agent
        rl_loop = row["model"](env, gamma=gamma, alpha=alpha, epsilon=epsilon, n_planning_steps=planning_steps)
        rl_loop.learn(num_episodes=train_episodes)

        # Get cumulative rewards
        episode_cumulative_rewards = rl_loop.cumulative_reward

        # Plot the results
        plt.plot(episode_cumulative_rewards, color=row["colour"], label=row["label"])

    plt.ylim(bottom=0, top=env_specs[environment_name]["ylim"])    # Set y-limits after all plots are generated
    plt.xlim(left=0, right=env_specs[environment_name]["xlim"])    # Set x-limits after all plots are generated
    plt.xlabel("Episode")
    plt.ylabel("Episode steps")
    plt.title(f"Episode steps for Dyna agent (gamma={gamma})")
    plt.legend()
    plt.show()


def run_2():

        # Run parameters
        env_specs = {
            "BlockingMaze": {
                "env": BlockingMaze,
                "train_episodes": 400,
                "ylim": 150,
                "xlim": 3000,
                "n_runs": 2,
            },
            "ShortcutMaze": {
                "env": ShortcutMaze,
                "train_episodes": 400,
                "ylim": 400,
                "xlim": 6000,
                "n_runs": 2,
            },
        }

        environment_name = "ShortcutMaze"
        gamma = 0.95
        epsilon = 0.1
        alpha = 0.5
        planning_steps = 50
        run_specs = {
            "model": [Dyna, DynaPlus],
            "colour": ["blue", "red"],
            "label": ["Dyna-Q", "Dyna-Q+"],
        }
        run_specs = {
            "model": [Dyna],
            "colour": ["blue"],
            "label": ["Dyna-Q"],
        }
        run_specs = pd.DataFrame(run_specs)

        for i, row in run_specs.iterrows():
            # Create the environment
            env = env_specs[environment_name]["env"]()

            # Initialise the trial
            trial = Trial(
                row["model"],
                env,
                sessions=env_specs[environment_name]["n_runs"],
                episodes_per_session=env_specs[environment_name]["train_episodes"],
                random_seeds=None
            )

            # Run the trial
            trial.run()

            # Plot the results
            trial.plot(color=row["colour"], show_std=False)

        plt.ylim(bottom=0, top=env_specs[environment_name]["ylim"])    # Set y-limits after all plots are generated
        plt.xlim(left=0, right=env_specs[environment_name]["xlim"])    # Set x-limits after all plots are generated
        plt.xlabel("Episode")
        plt.ylabel("Episode steps")
        plt.title(f"Episode steps for Dyna agent (gamma={gamma})")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    run_2()
