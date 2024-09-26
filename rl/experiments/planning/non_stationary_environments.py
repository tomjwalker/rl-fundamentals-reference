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
            "n_runs": 5,
        },
        "ShortcutMaze": {
            "env": ShortcutMaze,
            "train_episodes": 400,
            "ylim": 400,
            "xlim": 6000,
            "n_runs": 5,
        },
    }

    environment_name = "BlockingMaze"

    dyna_kwargs = {
        "gamma": 0.95,
        "epsilon": 0.1,
        "alpha": 0.5,
        "n_planning_steps": 50,
    }

    dyna_plus_kwargs = dyna_kwargs.copy()
    dyna_plus_kwargs["kappa"] = 0.001

    run_specs = {
        "model": [Dyna, DynaPlus],
        "colour": ["blue", "red"],
        "label": ["Dyna-Q", "Dyna-Q+"],
        "kwargs": [dyna_kwargs, dyna_plus_kwargs],
    }
    run_specs = pd.DataFrame(run_specs)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, row in run_specs.iterrows():
        # Get the environment class
        env_class = env_specs[environment_name]["env"]

        # Initialise the trial
        trial = Trial(
            agent_class=row["model"],
            environment_class=env_class,
            sessions=env_specs[environment_name]["n_runs"],
            episodes_per_session=env_specs[environment_name]["train_episodes"],
            random_seeds=None,
            **row["kwargs"],
        )

        # Run the trial
        trial.run()

        # Plot the results
        trial.plot(
            series_type="cumulative_rewards",
            color=row["colour"],
            show_std=False,
            std_alpha=0.1,
            ax=ax,
        )

    plt.ylim(bottom=0, top=env_specs[environment_name]["ylim"])    # Set y-limits after all plots are generated
    plt.xlim(left=0, right=env_specs[environment_name]["xlim"])    # Set x-limits after all plots are generated
    plt.title(f"Episode steps for Dyna agent (gamma={dyna_kwargs['gamma']})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run()
