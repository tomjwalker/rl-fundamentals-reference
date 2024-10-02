from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats


def plot_trial_results(
    rewards_testbed: pd.DataFrame,
    optimal_action_testbed: pd.DataFrame,
    params: Dict[str, float],
    ax: List[plt.Axes],
    color: str,
    show_individual_runs: bool = False,
    show_confidence_interval: bool = False
) -> None:
    """
    Plot the results of the trials for a given set of parameters.

    Args:
        rewards_testbed (pd.DataFrame): DataFrame containing rewards for each step and run.
        optimal_action_testbed (pd.DataFrame): DataFrame indicating if the optimal action was taken.
        params (Dict[str, float]): Parameters used in the experiment.
        ax (List[plt.Axes]): List of matplotlib axes to plot on.
        color (str): Color to use for plotting.
        show_individual_runs (bool): Whether to plot individual runs.
        show_confidence_interval (bool): Whether to show the confidence interval.
    """
    num_runs = rewards_testbed.shape[1]
    steps = rewards_testbed.shape[0]

    # Plot individual runs if specified
    alpha = 0.15
    if show_individual_runs:
        for i in range(num_runs):
            ax[0].plot(rewards_testbed.iloc[:, i], color=color, alpha=alpha)
            ax[1].plot(optimal_action_testbed.iloc[:, i] * 100, color=color, alpha=alpha)

    # Calculate mean
    mean_rewards = rewards_testbed.mean(axis=1)
    mean_optimal = optimal_action_testbed.mean(axis=1) * 100

    # Plot mean
    ax[0].plot(mean_rewards, color=color, linewidth=2, label=f"init={params['init']}")
    ax[1].plot(mean_optimal, color=color, linewidth=2, label=f"init={params['init']}")

    # Calculate and plot confidence interval if specified
    if show_confidence_interval:
        std_rewards = rewards_testbed.std(axis=1)
        ci_rewards = stats.t.interval(0.95, num_runs - 1, loc=mean_rewards, scale=std_rewards / np.sqrt(num_runs))

        std_optimal = optimal_action_testbed.std(axis=1) * 100
        ci_optimal = stats.t.interval(0.95, num_runs - 1, loc=mean_optimal, scale=std_optimal / np.sqrt(num_runs))

        ax[0].fill_between(range(steps), ci_rewards[0], ci_rewards[1], color=color, alpha=0.3)
        ax[1].fill_between(range(steps), ci_optimal[0], ci_optimal[1], color=color, alpha=0.3)
