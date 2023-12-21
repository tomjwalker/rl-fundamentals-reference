"""
TODO:
    - Not mega happy with this specific implementation (see policy update)
    - Unify SARSA, Expected SARSA, Q-learning with a Base class
    - Bring V/Q plots across from other repo
    - Unified plotting and experimentation of the three TD algorithms from pp. 132-133 of Sutton and Barto
"""


# Import custom argmax function
from utils.general import argmax

import gymnasium as gym
import numpy as np

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')


class ExpectedSarsa:

    def __init__(self, env, alpha=0.5, gamma=1.0, epsilon=0.1):
        self.name = "Expected Sarsa"
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_values = None
        self.policy = None
        self.episode_rewards = None
        self.reset()

    def reset(self):
        self.q_values = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.policy = np.ones_like(self.q_values) * self.epsilon / self.env.action_space.n
        self.episode_rewards = []  # Stores total reward for each episode

    def act(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return argmax(self.q_values[state, :])

    def train(self, num_episodes=500):

        for episode in range(num_episodes):

            # Initialise S
            state, _ = self.env.reset()

            # Initialise reward counter
            episode_reward = 0

            # Loop over each step of episode, until S is terminal
            done = False
            while not done:

                # Choose A from S using policy derived from Q (epsilon-greedy)
                action = self.act(state)

                # Take action A, observe R, S'
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Update Q(S, A), taking as target the expected-sarsa TD target (R + gamma * sum_a pi(a|S') * Q(S', a))
                target = reward + self.gamma * np.sum(self.policy[next_state, :] * self.q_values[next_state, :])
                self.q_values[state][action] += self.alpha * (target - self.q_values[state][action])

                # Update policy, which is epsilon-greedy with respect to the q-values
                self.policy[state] = self.epsilon / self.env.action_space.n
                best_action = np.argmax(self.q_values[state])
                self.policy[state][best_action] = 1 - self.epsilon + self.epsilon / self.env.action_space.n

                # S <- S', A <- A'
                state = next_state

                # Add reward to episode reward
                episode_reward += reward

                # If S is terminal, then episode is done (exit loop)
                done = terminated or truncated

            # Add episode reward to list
            self.episode_rewards.append(episode_reward)

            if episode % 100 == 0:
                print(f"Episode {episode}")
                print(f"Sum of episode rewards: {episode_reward}")
                print()


def plot_episode_rewards(episode_rewards, title):
    """
    Plot the episode rewards.
    """
    # Plot the episode rewards
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episode_rewards)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    # Set y-limits
    ax.set_ylim([-100, 0])
    ax.set_title(title)
    plt.show()


def run():

    # Run parameters
    train_episodes = 1000

    # Create the environment
    env = gym.make("CliffWalking-v0")
    rl_loop = ExpectedSarsa(env)
    rl_loop.train(num_episodes=train_episodes)

    # Plot the results
    plot_episode_rewards(rl_loop.episode_rewards, rl_loop.name)


if __name__ == "__main__":
    run()
