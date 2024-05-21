import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
matplotlib.use('TkAgg')


CWD = os.getcwd()
POLICY_DIR = os.path.join(CWD, "../.data", "value_iteration", "policy")
VALUE_DIR = os.path.join(CWD, "../.data", "value_iteration", "value")


def plot_policies():
    """
    A function which plots the following:
    - Policies:
        - π_0: a 20x20 grid of all actions = 0
        - π_1: policy_improvement_0.npy
        - π_2: policy_improvement_1.npy
        ...
        - π_4: policy_improvement_3.npy
    - Values: policy_evaluation_3.npy

    Plots a single figure, with a 2x3 subplot layout, with the following subplots:
    - π_0 | π_1 | π_2
    - π_3 | π_4 | V

    All policy plots are heatmaps of the policy. Use "seismic" colormap centered at 0 (white) to show +ve (red) and
    -ve (blue) actions

    The value plot is a heatmap of the value. Use "viridis" colormap to show the value.
    """

    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    policies = {}
    policies[0] = np.zeros((20, 20))
    for loop_num in [0, 5, 10, 100, 159]:
        policy_filepath = os.path.join(POLICY_DIR, f"value_iteration_{loop_num}.npy")
        policy = np.load(policy_filepath)
        policies[loop_num] = policy
    value_filepath = os.path.join(VALUE_DIR, f"value_iteration_{loop_num}.npy")
    value = np.load(value_filepath)

    # Plot the policies
    for i, (k, v) in enumerate(policies.items()):
        # i += 1
        if k == 159:
            # Heatmap of policy. Turn off the cell labels
            sns.heatmap(v, cmap="seismic", center=0, annot=False, cbar=False, ax=ax[1, 1])
            ax[1, 1].set_title(f"π_{k+1}")
            ax[1, 1].set_xlabel("Location 2")
            ax[1, 1].set_ylabel("Location 1")
            ax[1, 1].invert_yaxis()

            # Heatmap of value. Turn off the cell labels
            sns.heatmap(value, cmap="viridis", annot=False, cbar=False, ax=ax[1, 2])
            ax[1, 2].set_title(f"Value (π_{k+1})")
            ax[1, 2].invert_yaxis()
        else:
            # Heatmap of policy. Ensure that if the policy is all 0s, the heatmap is still centered at 0
            axes_row = i // 3
            axes_col = i % 3
            policy = policies[k]
            if np.all(policy == 0):
                # Plot all white heatmap
                sns.heatmap(policy, cmap="seismic", center=0, annot=False, fmt=".1f", cbar=False, ax=ax[axes_row, axes_col])
            else:
                sns.heatmap(policy, cmap="seismic", center=0, annot=False, fmt=".1f", cbar=False, ax=ax[axes_row, axes_col])
            ax[axes_row, axes_col].set_title(f"π {k+1}")
            ax[axes_row, axes_col].set_xlabel("Location 2")
            ax[axes_row, axes_col].set_ylabel("Location 1")
            ax[axes_row, axes_col].invert_yaxis()

    # On the right of the first row of subplots, add a colorbar for the policy heatmaps
    # Discretise the colorbar into 11 bins
    cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.35])    # [left, bottom, width, height]
    fig.colorbar(ax[0, 1].collections[0], cax=cbar_ax)

    # On the right of the second row of subplots, add a colorbar for the value heatmap
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.35])
    fig.colorbar(ax[1, 2].collections[0], cax=cbar_ax)

    plt.show()


def main():
    plot_policies()


if __name__ == "__main__":
    main()
