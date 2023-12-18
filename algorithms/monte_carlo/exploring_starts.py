from utils.general import argmax

import gymnasium as gym
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('TkAgg')


env = gym.make("Blackjack-v1", sab=True)    # `sab` means rules following Sutton and Barto


def _is_subelement_present(subelement, my_list):
    """
    Helps check if a subelement is present in a list of tuples. Used to check if state has already been seen.

    Simple example:
    _is_subelement_present((1, 2), [(1, 2, 3), (4, 5, 6)])
        True
    """
    for tpl in my_list:
        if subelement == tpl[:len(subelement)]:
            return True
    return False


class MCControl:

    def __init__(self, env):
        self.env = env

        self.q_values = None
        self.policy = None
        self.returns = None
        self.reset()

    def _init_policy(self, state_shape):
        """
        Use the policy initialisation from Sutton and Barto, pp. 93:
        - If player sum == 20 or 21, stick
        - Otherwise, hit
        """
        self.policy = np.ones(state_shape, dtype=np.int8)    # 0 = stick, 1 = hit
        self.policy[19:, :, :] = 0
        # self.policy[19:21, :, :] = 1
        # print(self.policy[:, :, 0])
        # print(self.policy[:, :, 1])

    def reset(self):
        # Get env shape
        state_shape = ()
        for space in self.env.observation_space:
            state_shape += (space.n,)

        # Initialise q-values, policy, and returns
        state_and_action_shape = state_shape + (self.env.action_space.n,)
        self.q_values = np.zeros(state_and_action_shape)
        self._init_policy(state_shape)
        # Returns is a tensor same shape as q-values, but with each element being a list of returns
        self.returns = np.empty(state_and_action_shape, dtype=object)
        for index in np.ndindex(state_and_action_shape):
            self.returns[index] = []

    def act(self, state):
        """Greedy policy"""
        return argmax(self.q_values[state])

    def train(self, num_episodes=10000, gamma=1.0):

        for episode_idx in range(num_episodes):

            # Print progress
            if episode_idx % 1000 == 0:
                print(f"Episode {episode_idx}")

            # Exploring start selection of S_0 and A_0
            state, info = self.env.reset()    # S_0
            action = np.random.randint(0, self.env.action_space.n)    # A_0: choice of {0, 1}

            # Generate an episode
            episode = []
            while True:
                next_state, reward, terminated, truncated, info = self.env.step(action)
                episode.append((state, action, reward))

                done = terminated or truncated
                if done:
                    break

                state = next_state
                action = self.act(state)

            # if len(episode) > 5:
            #     print("interesting!")

            # Once the episode is complete (the `while True` loop has broken), update the q-values and policy
            # Loop through the episode in reverse order, updating the q-values
            g = 0
            for t, (state, action, reward) in enumerate(reversed(episode)):
                g = gamma * g + reward

                # If the S_t, A_t pair has been seen before, continue.
                if _is_subelement_present((state, action), episode[:len(episode) - t - 1]):
                    continue

                # Add the return to the list of returns for this state-action pair
                self.returns[state][action].append(g)

                # Update the q-value for this state-action pair
                self.q_values[state][action] = np.mean(self.returns[state][action])

                # Update the policy
                self.policy[state] = argmax(self.q_values[state][:])


def get_3d_plot(mc_control, usable_ace=True, fig=None, subplot=111):
    # Get (state) values for a usable ace policy
    if usable_ace:
        values_usable_ace = np.max(mc_control.q_values[:, :, 1, :], axis=2)
    else:
        values_usable_ace = np.max(mc_control.q_values[:, :, 0, :], axis=2)

    # If ax is not provided, create a new 3D axis
    if fig is None:
        fig = plt.figure()

    # Clip the values_usable_ace axes to 12-21 and 1-10
    values_usable_ace = values_usable_ace[12:22, 1:11]

    ax = fig.add_subplot(subplot, projection='3d')

    # Determine meshgrid from state space shape
    x_start = 10
    y_start = 12
    x = np.arange(x_start, x_start + values_usable_ace.shape[1])
    y = np.arange(y_start, y_start + values_usable_ace.shape[0])
    x, y = np.meshgrid(x, y)

    # Use the plot_surface method
    surface = ax.plot_surface(x, y, values_usable_ace, cmap="viridis")

    # Limit z-axis to -1 to 1
    ax.set_zlim(-1, 1)

    return fig, ax


def plot_policy(mc_control):

    usable_ace = mc_control.policy[11:22, :, 0]
    usable_ace = pd.DataFrame(usable_ace)
    usable_ace.index = np.arange(11, 22)
    usable_ace.columns = np.arange(1, 11)
    no_usable_ace = mc_control.policy[11:22, :, 1]
    no_usable_ace = pd.DataFrame(no_usable_ace)
    no_usable_ace.index = np.arange(11, 22)
    no_usable_ace.columns = np.arange(1, 11)

    fig, ax = plt.subplots(1, 2)


def run():

    TRAIN_EPISODES = 100000

    mc_control = MCControl(env)
    mc_control.train(num_episodes=TRAIN_EPISODES)

    # Plot the state value functions - with and without usable ace
    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig, ax_0 = get_3d_plot(mc_control, usable_ace=True, fig=fig, subplot=121)
    ax_0.set_title("Usable ace")
    ax_0.set_xlabel("Dealer showing")
    ax_0.set_ylabel("Player sum")
    ax_0.set_zlabel("Value")
    fig, ax_1 = get_3d_plot(mc_control, usable_ace=False, fig=fig, subplot=122)
    ax_1.set_title("No usable ace")
    ax_1.set_xlabel("Dealer showing")
    ax_1.set_ylabel("Player sum")
    ax_1.set_zlabel("Value")

    plt.show()


if __name__ == "__main__":
    run()
