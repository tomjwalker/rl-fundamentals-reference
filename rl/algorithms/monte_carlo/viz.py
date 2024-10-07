import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap

matplotlib.use('TkAgg')


# TODO: https://seaborn.pydata.org/tutorial/color_palettes.html#diverging-color-palettes


DEALER_MIN = 1
DEALER_MAX = 10
PLAYER_MIN_POLICY = 11
PLAYER_MIN_VALUE = 12
PLAYER_MAX = 21


def _argmax_last_tie(array):
    # Reverse the array along the last axis
    reversed_array = array[..., ::-1]

    # Apply argmax on the reversed array along the last axis
    # This gives us the index of the maximum value in the reversed array
    reversed_argmax = np.argmax(reversed_array, axis=-1)

    # Convert the index from the reversed array to the original array
    original_argmax = array.shape[-1] - 1 - reversed_argmax

    return original_argmax


def _get_policy_for_agent(mc_agent, usable_ace):
    """
    Get the policy for the given agent and ace usability.
    Returns a 2D array representing the policy.
    """
    if mc_agent.name == "MC Exploring Starts" or mc_agent.name == "MC Off-Policy":
        # For deterministic policies
        policy = mc_agent.policy.action_map[PLAYER_MIN_POLICY:PLAYER_MAX+1, DEALER_MIN:DEALER_MAX+1, 1 if usable_ace
                                                                                                  else 0]
    elif mc_agent.name == "MC On-Policy":
        # For stochastic policies, return the action with the highest probability
        if usable_ace:
            q_vals = mc_agent.q_values.stats[PLAYER_MIN_POLICY:PLAYER_MAX + 1, DEALER_MIN:DEALER_MAX + 1, 1, :]
        else:
            q_vals = mc_agent.q_values.stats[PLAYER_MIN_POLICY:PLAYER_MAX + 1, DEALER_MIN:DEALER_MAX + 1, 0, :]
        policy = _argmax_last_tie(q_vals)

    else:
        raise ValueError(f"Unknown agent type: {mc_agent.name}")
    return policy


def _generate_3d_value_ax(mc_control, usable_ace=True, fig=None, subplot=111):
    """
    Generate a 3D surface plot of state values.
    """
    values = np.max(mc_control.q_values.values[:, :, 1 if usable_ace else 0, :], axis=2)
    title = "State value: usable ace" if usable_ace else "State value: no usable ace"

    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(subplot, projection='3d')

    values = values[PLAYER_MIN_VALUE:PLAYER_MAX+1, DEALER_MIN:DEALER_MAX+1]

    player_count = np.arange(PLAYER_MIN_VALUE, PLAYER_MAX+1)
    dealer_count = np.arange(DEALER_MIN, DEALER_MAX+1)
    x, y = np.meshgrid(dealer_count, player_count)

    # Create the seaborn colormap
    cmap = sns.color_palette("vlag", as_cmap=True)

    _ = ax.plot_surface(x, y, values, cmap=cmap)

    ax.set_title(title)
    ax.set_xlabel("Dealer showing")
    ax.set_ylabel("Player sum")
    ax.set_zlabel("Value")
    ax.set_zlim(-1, 1)

    return fig, ax


def _generate_policy_ax(mc_agent, usable_ace=True, fig=None, subplot=111):
    """
    Generate a heatmap of the policy.
    """
    policy = _get_policy_for_agent(mc_agent, usable_ace)
    title = "Policy: usable ace" if usable_ace else "Policy: no usable ace"

    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(subplot)

    policy = pd.DataFrame(policy)
    policy.index = np.arange(PLAYER_MIN_POLICY, PLAYER_MAX+1)
    policy.columns = np.arange(DEALER_MIN, DEALER_MAX+1)

    # Create a categorical colormap using 'tab10'
    cmap = ListedColormap(plt.get_cmap('tab10').colors[:2])

    # Create the heatmap with the categorical colormap
    sns.heatmap(policy, cmap=cmap, annot=False, fmt="d", ax=ax, cbar=False)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Dealer showing")
    ax.set_ylabel("Player sum")

    # Create a custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0), markersize=10, label='Stick'),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(1), markersize=10, label='Hit')]
    ax.legend(handles=handles, title='Action', bbox_to_anchor=(1.05, 1), loc='upper left')

    return fig, ax


def plot_results(mc_control):
    """
    Plot state value functions and policy as a 2x2 grid of plots.
    """
    fig = plt.figure(figsize=plt.figaspect(1))
    fig, ax_0 = _generate_3d_value_ax(mc_control, usable_ace=True, fig=fig, subplot=221)
    fig, ax_1 = _generate_3d_value_ax(mc_control, usable_ace=False, fig=fig, subplot=222)
    fig, ax_2 = _generate_policy_ax(mc_control, usable_ace=True, fig=fig, subplot=223)
    fig, ax_3 = _generate_policy_ax(mc_control, usable_ace=False, fig=fig, subplot=224)
    fig.suptitle(mc_control.name)

    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming `mc_control` is defined and PLAYER_MIN_VALUE, PLAYER_MAX, DEALER_MIN, DEALER_MAX are set appropriately
# plot_results(mc_control)
