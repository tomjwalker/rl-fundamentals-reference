"""
TODO:
    - Figure out why plots not exactly the same as Sutton and Barto
"""


from algorithms.monte_carlo.viz import plot_results
from utils.general import argmax

import gymnasium as gym
import numpy as np

import matplotlib

matplotlib.use('TkAgg')


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

        self.name = "MC Exploring Starts"    # For plotting

        self.q_values = None
        self.policy = None
        self.returns = None
        self.reset()

    def _init_policy(self, state_shape):
        """
        Use the target_policy initialisation from Sutton and Barto, pp. 93:
        - If player sum == 20 or 21, stick
        - Otherwise, hit
        """
        self.policy = np.ones(state_shape, dtype=np.int8)    # 0 = stick, 1 = hit
        self.policy[19:, :, :] = 0
        # self.target_policy[19:21, :, :] = 1
        # print(self.target_policy[:, :, 0])
        # print(self.target_policy[:, :, 1])

    def reset(self):
        # Get env shape
        state_shape = ()
        for space in self.env.observation_space:
            state_shape += (space.n,)

        # Initialise q-values, target_policy, and returns
        state_and_action_shape = state_shape + (self.env.action_space.n,)
        self.q_values = np.zeros(state_and_action_shape)
        self._init_policy(state_shape)
        # Returns is a tensor same shape as q-values, but with each element being a list of returns
        self.returns = np.empty(state_and_action_shape, dtype=object)
        for index in np.ndindex(state_and_action_shape):
            self.returns[index] = []

    def act(self, state):
        """Greedy target_policy"""
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

            # Once the episode is complete (the `while True` loop has broken), update the q-values and target_policy
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

                # Update the target_policy
                self.policy[state] = argmax(self.q_values[state][:])


def run():

    # Run parameters
    train_episodes = 10000

    # Instantiate and train the agent
    env = gym.make("Blackjack-v1", sab=True)  # `sab` means rules following Sutton and Barto
    mc_control = MCControl(env)
    mc_control.train(num_episodes=train_episodes)

    # Plot the results
    plot_results(mc_control)


if __name__ == "__main__":
    run()
