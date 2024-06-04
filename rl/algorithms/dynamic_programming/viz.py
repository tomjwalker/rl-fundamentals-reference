import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("TkAgg")


def plot_value_function_3d(value, figure=None):
    if figure is None:
        figure = plt.figure(figsize=(12, 6))

    ax = figure.add_subplot(122, projection="3d")
    x = np.arange(value.shape[0])
    y = np.arange(value.shape[1])
    x, y = np.meshgrid(x, y)
    z = value.T  # Transpose so that x and y correspond to the correct axes

    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel("Number of cars at location 1")
    ax.set_ylabel("Number of cars at location 2")
    ax.set_zlabel("Value")
    ax.set_title("Value Function")
    return figure, ax


def plot_policy(policy, figure=None):
    if figure is None:
        figure = plt.figure(figsize=(12, 6))

    ax = figure.add_subplot(121)
    sns.heatmap(policy, cmap="seismic", center=0, annot=True, fmt=".1f", cbar=False, square=True, ax=ax)
    ax.invert_yaxis()  # So the y-axis runs from 0 (bottom) to max_cars (top)
    ax.set_title("Policy")
    ax.set_xlabel("Number of cars at location 2")
    ax.set_ylabel("Number of cars at location 1")
    return figure, ax


def plot_policy_and_value(policy, value):
    figure = plt.figure(figsize=(15, 7.5))

    figure, ax1 = plot_policy(policy, figure)
    figure, ax2 = plot_value_function_3d(value, figure)

    # Adjust layout to prevent clipping
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    plt.show()
