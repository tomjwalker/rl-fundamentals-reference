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
import argparse
from typing import Optional

matplotlib.use('TkAgg')


# TODO: currently plot for BlockingMaze not matching S&B - dyna-q (vanilla) sub-par


def run(environment_name: str, train_episodes: int, n_runs: int, agents: Optional[list] = None) -> None:
    """
    Run the Dyna and Dyna-Q+ experiments in the specified gridworld environment.

    Args:
        environment_name (str): The name of the environment to run (BlockingMaze or ShortcutMaze).
        train_episodes (int): The number of episodes to train the agent.
        n_runs (int): The number of runs for each agent.
        agents (Optional[list]): List of agents to run (e.g., ["Dyna", "DynaPlus"]).
    """
    # Run parameters
    env_specs = {
        "BlockingMaze": {
            "env": BlockingMaze,
            "ylim": 150,
            "xlim": 3000,
        },
        "ShortcutMaze": {
            "env": ShortcutMaze,
            "ylim": 400,
            "xlim": 6000,
        },
    }

    if environment_name not in env_specs:
        raise ValueError(f"Invalid environment name: {environment_name}. Choose from: {list(env_specs.keys())}")

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

    if agents:
        run_specs = run_specs[run_specs["label"].isin(agents)]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, row in run_specs.iterrows():
        # Get the environment class
        env_class = env_specs[environment_name]["env"]

        # Initialise the trial
        trial = Trial(
            agent_class=row["model"],
            environment_class=env_class,
            sessions=n_runs,
            episodes_per_session=train_episodes,
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

    plt.ylim(bottom=0, top=env_specs[environment_name]["ylim"])  # Set y-limits after all plots are generated
    plt.xlim(left=0, right=env_specs[environment_name]["xlim"])  # Set x-limits after all plots are generated
    plt.title(f"Episode steps for Dyna agent (gamma={dyna_kwargs['gamma']})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dyna agents on gridworld environments and plot the results.")
    parser.add_argument(
        '--environment_name',
        type=str,
        default="BlockingMaze",
        help="The name of the environment to run (BlockingMaze or ShortcutMaze)."
    )
    parser.add_argument(
        '--train_episodes',
        type=int,
        default=400,
        help="The number of episodes to train the agent."
    )
    parser.add_argument(
        '--n_runs',
        type=int,
        default=5,
        help="The number of runs for each agent."
    )
    parser.add_argument(
        '--agents',
        type=str,
        nargs='+',
        choices=["Dyna-Q", "Dyna-Q+"],
        help="Specify which agents to run (Dyna-Q, Dyna-Q+). If not specified, both will be run."
    )
    args = parser.parse_args()

    run(
        environment_name=args.environment_name,
        train_episodes=args.train_episodes,
        n_runs=args.n_runs,
        agents=args.agents
    )
