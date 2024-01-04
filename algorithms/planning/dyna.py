# Import custom argmax function
import pandas as pd

from utils.general import argmax

# Import relevant gridworld environment (local implementation of gridworld pp165 of Sutton and Barto (2018))
from environment.planning_maze import Maze
#
# import gymnasium as gym
import numpy as np
from collections import defaultdict    # For model, with default value of empty list
import random    # For random choice of state and action from model during planning
random.seed(42)

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')


class Dyna:

    def __init__(self, env, alpha=0.5, gamma=1.0, epsilon=0.1, n_planning_steps=5):
        self.name = "Dyna"
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning_steps = n_planning_steps

        # Used for plotting cum reward vs training step plots (see Sutton and Barto, 2018, pp 167)
        self.cumulative_reward = []

        self.q_values = None
        self.model = None
        self.policy = None
        self.episode_rewards = None
        self.episode_steps = None
        self.reset()

    def reset(self):
        self.q_values = np.zeros(shape=(self.env.observation_space.n, self.env.action_space.n))
        self.policy = np.zeros_like(self.q_values)
        self.episode_rewards = []  # Stores total reward for each episode
        self.episode_steps = []  # Stores total steps for each episode

        # Model is a dictionary of lists, where each list contains tuples of (reward, next_state), and the key is the
        # #(state, action) pair
        self.model = defaultdict(list)

    def update_cumulative_reward(self, reward):
        if len(self.cumulative_reward) == 0:
            self.cumulative_reward.append(reward)
        else:
            self.cumulative_reward.append(reward + self.cumulative_reward[-1])

    def act(self, state):
        """Given a state, return an action according to the epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return argmax(self.q_values[state, :])

    def train(self, num_episodes=500):

        for episode in range(num_episodes):

            # Initialise S (**a**)
            state, _ = self.env.reset()

            # Initialise logging variables
            episode_reward = 0
            episode_steps = 0

            # Loop over each step of episode, until S is terminal
            done = False
            while not done:

                # Choose A from S using policy derived from Q (epsilon-greedy) (**b**)
                action = self.act(state)

                # Take action A, observe R, S' (**c**)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Update Q(S, A) (**d**)
                best_next_action = argmax(self.q_values[next_state, :])
                td_target = reward + self.gamma * self.q_values[next_state, best_next_action]
                self.q_values[state][action] += self.alpha * (td_target - self.q_values[state][action])

                # Update logs
                episode_reward += reward
                episode_steps += 1
                self.update_cumulative_reward(reward)

                # Update model (**e**).
                # Model is a dictionary: {(state, action): [(reward, next_state)_1, (reward, next_state)_2]}
                # If (reward, next_state) is not in the list, add it, otherwise do nothing
                if (state, action) not in self.model:
                    self.model[(state, action)].append((reward, next_state))
                elif (reward, next_state) not in self.model[(state, action)]:
                    self.model[(state, action)].append((reward, next_state))

                # Loop for n planning steps, and perform planning updates (**f**)
                for _ in range(self.n_planning_steps):

                    # Choose a random, previously observed state and action (**f.i, f.ii**)
                    (state, action) = random.choice(list(self.model.keys()))

                    # TODO: check want to retain this line
                    # Get reward and next state from model (N.B. unlike S&B, this accounts for stochasticity in the
                    # environment) (**f.iii**)
                    # Make sure next state from learning is not mixed up with next state from planning (for S <- S')
                    (reward, next_state_plan) = random.choice(self.model[(state, action)])

                    # Update Q(S, A), taking as target the q-learning TD target (R + gamma * max_a Q(S', a)) (**f.iv**)
                    td_target = reward + self.gamma * np.max(self.q_values[next_state_plan, :])
                    self.q_values[state][action] += self.alpha * (td_target - self.q_values[state][action])

                # S <- S'
                state = next_state

                # If S is terminal, then episode is done (exit loop)
                done = terminated or truncated

            # Update logs
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(episode_steps)

            if episode % (num_episodes // 10) == 0:
                print(f"Episode {episode}")
                print(f"Cumulative reward: {self.cumulative_reward[-1]}")
                print()


def run():

    # Run parameters
    train_episodes = 50
    gamma = 0.95
    alpha = 0.1
    epsilon = 0.1
    run_specs = {
        "planning steps": [0, 5, 50],
        "colour": ["blue", "green", "red"],
        "label": ["0 planning steps (direct RL)", "5 planning steps", "50 planning steps"],
    }
    run_specs = pd.DataFrame(run_specs)

    for i, row in run_specs.iterrows():

        # Create the environment
        env = Maze()

        # Create and train the agent
        rl_loop = Dyna(env, gamma=gamma, alpha=alpha, epsilon=epsilon, n_planning_steps=row["planning steps"])
        rl_loop.train(num_episodes=train_episodes)

        # Plot the results
        plt.plot(rl_loop.episode_steps, color=row["colour"], label=row["label"])

    plt.ylim(bottom=0, top=800)    # Set y-limits after all plots are generated
    plt.xlabel("Episode")
    plt.ylabel("Episode steps")
    plt.title(f"Episode steps for Dyna agent (gamma={gamma})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run()
