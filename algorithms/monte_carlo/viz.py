import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def _generate_3d_value_ax(mc_control, usable_ace=True, fig=None, subplot=111):
    # Get (state) values for a usable ace target_policy
    if usable_ace:
        values = np.max(mc_control.q_values[:, :, 1, :], axis=2)
        title = "State value: usable ace"
    else:
        values = np.max(mc_control.q_values[:, :, 0, :], axis=2)
        title = "State value: no usable ace"

    # If ax is not provided, create a new 3D axis
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(subplot, projection='3d')

    # Clip the values_usable_ace axes to 12-21 and 1-10
    values = values[12:22, 1:11]

    # Determine meshgrid from state space shape
    x_start = 10
    y_start = 12
    x = np.arange(x_start, x_start + values.shape[1])
    y = np.arange(y_start, y_start + values.shape[0])
    x, y = np.meshgrid(x, y)

    # Use the plot_surface method
    _ = ax.plot_surface(x, y, values, cmap="viridis")

    ax.set_title(title)
    ax.set_xlabel("Dealer showing")
    ax.set_ylabel("Player sum")
    ax.set_zlabel("Value")
    ax.set_zlim(-1, 1)

    return fig, ax


def _generate_policy_ax(mc_agent, usable_ace=True, fig=None, subplot=111):

    # TODO: better as a method of a BaseAgent class, allowing for different specific plot methods for different agents?

    # Based on the type of agent, need to treat the target_policy differently
    if mc_agent.name == "MC Exploring Starts":
        # For exploring starts, target_policy is deterministic (π(s)) so the array is 3D with dimensions following the
        # state, and cells containing the action to take as an integer (0 = stick, 1 = hit)
        if usable_ace:
            policy = mc_agent.target_policy[11:22, :, 1]
            title = "Policy: usable ace"
        else:
            policy = mc_agent.target_policy[11:22, :, 0]
            title = "Policy: no usable ace"
    elif mc_agent.name == "MC On-Policy":
        # Other agents have stochastic policies (π(a|s)) so the array is 4D with dimensions following the state and
        # action, and cells containing the probability of taking that action from that state
        # For plotting, need 2D arrays (one each for usable ace and no usable ace). Arrange so cell values are the most
        # likely action to take from that state (i.e. argmax)
        if usable_ace:
            policy = np.argmax(mc_agent.target_policy[11:22, :, 1, :], axis=2)
            title = "Policy: usable ace"
        else:
            policy = np.argmax(mc_agent.target_policy[11:22, :, 0, :], axis=2)
            title = "Policy: no usable ace"
    elif mc_agent.name == "MC Off-Policy":
        # For off-policy, target_policy is deterministic (π(s)) so the array is 3D with dimensions following the
        # state, and cells containing the action to take as an integer (0 = stick, 1 = hit)
        if usable_ace:
            policy = mc_agent.target_policy[11:22, :, 1]
            title = "Policy: usable ace"
        else:
            policy = mc_agent.target_policy[11:22, :, 0]
            title = "Policy: no usable ace"

    # If ax is not provided, create a new 3D axis
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(subplot)

    policy = pd.DataFrame(policy)
    policy.index = np.arange(11, 22)
    policy.columns = np.arange(1, 12)

    # Plot the target_policy. Use "viridis" colormap. Subplots: target_policy with usable ace, target_policy without usable ace
    sns.heatmap(policy, cmap="viridis", annot=False, fmt="d", ax=ax)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Dealer showing")
    ax.set_ylabel("Player sum")

    return fig, ax


def plot_results(mc_control):
    """
    Uses _generate_3d_value_ax and _generate_policy_ax to plot the state value functions and target_policy, as a 2x2 grid of
    plots.

    Upper row: state value functions (with and without usable ace)
    Lower row: target_policy (with and without usable ace)
    """
    fig = plt.figure(figsize=plt.figaspect(1))
    fig, ax_0 = _generate_3d_value_ax(mc_control, usable_ace=True, fig=fig, subplot=221)
    fig, ax_1 = _generate_3d_value_ax(mc_control, usable_ace=False, fig=fig, subplot=222)
    fig, ax_2 = _generate_policy_ax(mc_control, usable_ace=True, fig=fig, subplot=223)
    _, _ = _generate_policy_ax(mc_control, usable_ace=False, fig=fig, subplot=224)
    fig.suptitle(mc_control.name)

    plt.tight_layout()
    plt.show()
