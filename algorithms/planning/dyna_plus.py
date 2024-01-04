# Import custom argmax function
import pandas as pd

from utils.general import argmax
from dyna import Dyna    # DynaPlus inherits from Dyna, so we can reuse a lot of the same code from Dyna

# Import relevant gridworld environment (local implementation of blocking and shortcut maze gridworlds, pp167 of Sutton
# and Barto (2018))
# from environment.planning_maze import Maze
from environment.blocking_maze import BlockingMaze
#
# import gymnasium as gym
import numpy as np
from collections import defaultdict    # For model, with default value of empty list
import random    # For random choice of state and action from model during planning
random.seed(42)

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')


class DynaPlus(Dyna):

    def __init__(self, env, alpha=0.5, gamma=1.0, epsilon=0.1, n_planning_steps=5, kappa=0.001):
        super().__init__(env, alpha, gamma, epsilon, n_planning_steps)
        self.name = "Dyna+"
        self.kappa = kappa

        # self.time_since_last_encounter is a dictionary with the same keys as self.model, and values of the number
        # of timesteps since the (S, A) tuple was last encountered.
        # If the state-action pair has never been encountered, the value is 0 (ensuring consistent dtype, and equal
        # initial values for all state-action pairs)
        # N.B. this is initialised outside the reset() method, as the total number of steps taken is not reset when
        # the environment is reset at the end of an episode
        self.time_since_last_encountered = self.model.copy()
        for key in self.time_since_last_encountered.keys():
            self.time_since_last_encountered[key] = 0

    def reset(self):
        super().reset()

        # Per footnote on pp 168, model is initialised with (reward=0, next_state=state) for all state-action pairs
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                self.model[(state, action)] = (0, state)

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

                # TODO: temp debug check
                if self.env.total_steps == 999:
                    print("debug")

                # Update Q(S, A) (**d**)
                best_next_action = argmax(self.q_values[next_state, :])
                td_target = reward + self.gamma * self.q_values[next_state, best_next_action]
                self.q_values[state][action] += self.alpha * (td_target - self.q_values[state][action])

                # Update logs
                episode_reward += reward
                episode_steps += 1
                self.update_cumulative_reward(reward)

                # Update model (**e**).
                # TODO: NB, values are single (reward, next_state) tuples, not lists of tuples unlike current Dyna
                # Model is a dictionary: {(state, action): (reward, next_state)}. Initialised with (reward=0,
                # next_state=state), so replace this with actual values from environment
                self.model[(state, action)] = (reward, next_state)
                # Update time since last encountered
                # i. Increment all values by 1
                for key in self.time_since_last_encountered.keys():
                    self.time_since_last_encountered[key] += 1
                # ii. Set time since last encountered for current state-action pair to 0
                self.time_since_last_encountered[(state, action)] = 0

                # Loop for n planning steps, and perform planning updates (**f**)
                for _ in range(self.n_planning_steps):

                    # Choose a random, previously observed state and action (**f.i, f.ii**)
                    (state, action) = random.choice(list(self.model.keys()))

                    # TODO: different from Dyna (no values as lists)
                    # Get reward and next state from model (N.B. unlike S&B, this accounts for stochasticity in the
                    # environment) (**f.iii**)
                    # Make sure next state from learning is not mixed up with next state from planning (for S <- S')
                    (reward, next_state_plan) = self.model[(state, action)]
                    time_since_last_encountered = self.time_since_last_encountered[(state, action)]

                    # Update Q(S, A), taking as target the q-learning TD target **with Dyna-Q+** additional term:
                    # TD_target = R + gamma * max_a Q(S', a) + kappa * sqrt(time_since_last_encountered) (**f.iv**)
                    best_next_action = argmax(self.q_values[next_state_plan, :])
                    td_target = reward + self.gamma * self.q_values[next_state_plan, best_next_action] + \
                        self.kappa * np.sqrt(time_since_last_encountered)
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
    train_episodes = 3000
    gamma = 0.95
    epsilon = 0.1
    alpha = 0.5
    planning_steps = 5
    run_specs = {
        "model": [Dyna, DynaPlus],
        "colour": ["blue", "red"],
        "label": ["Dyna-Q", "Dyna-Q+"],
    }
    run_specs = pd.DataFrame(run_specs)

    for i, row in run_specs.iterrows():

        # Create the environment
        env = BlockingMaze()

        # Create and train the agent
        rl_loop = row["model"](env, gamma=gamma, alpha=alpha, epsilon=epsilon, n_planning_steps=planning_steps)
        rl_loop.train(num_episodes=train_episodes)

        # Get cumulative rewards
        episode_cumulative_rewards = rl_loop.cumulative_reward

        # Plot the results
        plt.plot(episode_cumulative_rewards, color=row["colour"], label=row["label"])

    plt.ylim(bottom=0, top=150)    # Set y-limits after all plots are generated
    plt.xlim(left=0, right=3000)    # Set x-limits after all plots are generated
    plt.xlabel("Episode")
    plt.ylabel("Episode steps")
    plt.title(f"Episode steps for Dyna agent (gamma={gamma})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run()
