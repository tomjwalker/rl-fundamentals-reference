"""Draw illustrative violin plots for 4 k-armed bandit problems, with different means and variances."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal

import matplotlib
matplotlib.use('TkAgg')


def plot_violin_plots():
    # Set the random seed
    random_seed = 0
    np.random.seed(random_seed)

    # Initialise the k-armed bandits
    num_runs = 500  # Final version: 2000
    k = 2
    k_mean = 0
    k_std = 1
    bandit_std = 1

    # Generate the bandit means and standard deviations
    bandit_means = np.random.normal(k_mean, k_std, k)
    bandit_stds = np.random.normal(bandit_std, 0.5, k)

    # Generate the rewards for each bandit
    rewards = np.random.normal(bandit_means, bandit_stds, (num_runs, k))

    # Plot the violin plots
    plt.figure(figsize=(9, 7))
    sns.violinplot(data=rewards)
    plt.xlabel("Action")
    plt.ylabel("Reward")
    plt.title("Reward distributions for 4-Armed Bandits")
    plt.show()


"""
Ok, now I want 2 subplots - each axis contains a 2-distribution violin plots for a 2-armed bandit. 
On the LHS axis, action 0 has a greater mean than action 1. Vice versa for the RHS axis.
This is to illustrate a contextual bandit problem.
"""

def plot_contextual_violin_plots():

    # Set the random seed
    random_seed = 0
    np.random.seed(random_seed)

    # Initialise the k-armed bandits
    num_runs = 500  # Final version: 2000
    k = 2
    k_mean = 0
    k_std = 1
    bandit_std = 1

    # Generate the bandit means and standard deviations
    bandit_means = np.random.normal(k_mean, k_std, k)
    bandit_stds = np.random.normal(bandit_std, 0.5, k)

    # Generate the rewards for each bandit
    rewards_0 = np.random.normal([1, 0.5], [0.3, 0.1], (num_runs, k))
    rewards_1 = np.random.normal([0.5, 1], [0.3, 0.4], (num_runs, k))
    rewards = [rewards_0, rewards_1]

    # Plot the violin plots
    fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
    for i in range(2):
        sns.violinplot(data=rewards[i], ax=ax[i])
        ax[i].set_xlabel("Action")
        ax[i].set_ylabel("Reward")
        ax[i].set_title(f"Reward distributions for 2-Armed Bandit {i+1}")
    plt.show()


"""
Now I want to generate a 2x1 subplot showing two different bivariate normal distributions (to illustrate
p(s', r | s, a) for two different (s, a) pairs).

Use the same z-limits so the two bivariate normal distributions look different.

Use scipy.stats.multivariate_normal to generate the bivariate normal distributions.

Use ax.plot_surface to plot the bivariate normal distributions.
"""

def plot_bivariate_normal():
    # Set the random seed
    random_seed = 0
    np.random.seed(random_seed)

    # Generate the bivariate normal distributions
    means = np.array([[0, 0], [0, 0]])
    covs = np.array([[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]]])
    bivariate_normals = [multivariate_normal(mean=means[i], cov=covs[i]) for i in range(2)]

    # Plot the bivariate normal distributions
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(121, projection="3d")
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    x, y = np.meshgrid(x, y)
    z = bivariate_normals[0].pdf(np.dstack((x, y)))
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel("s'")
    ax.set_ylabel("r")
    ax.set_zlabel("p(s', r | s_0, a_0)")
    ax.set_title("Dynamics function for p(s', r | s_0, a_0)")

    ax = fig.add_subplot(122, projection="3d")
    z = bivariate_normals[1].pdf(np.dstack((x, y)))
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel("s'")
    ax.set_ylabel("r")
    ax.set_zlabel("p(s', r | s_1, a_1)")
    ax.set_title("Dynamics function for p(s', r | s_1, a_1)")

    plt.show()


if __name__ == "__main__":
    # plot_violin_plots()
    # plot_contextual_violin_plots()
    plot_bivariate_normal()
    print("done!")
