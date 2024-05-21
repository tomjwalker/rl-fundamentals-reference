import gymnasium as gym
from rl.algorithms.common.td_agent import TemporalDifferenceAgent

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')


class QLearning(TemporalDifferenceAgent):

    def __init__(self, env, alpha=0.5, gamma=1.0, epsilon=0.1, random_seed=None):
        super().__init__(env, gamma, alpha, epsilon, random_seed)
        self.name = "Q-Learning"

    def learn(self, num_episodes=500):

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

                # Update Q(S, A), taking as target the q-learning TD target (R + gamma * max_a Q(S', a))
                td_target = reward + self.gamma * self.q_values.get(next_state, self.q_values.get_max_action(next_state))
                td_error = td_target - self.q_values.get(state, action)
                new_value = self.q_values.get(state, action) + self.alpha * td_error
                self.q_values.update(state, action, new_value)

                # S <- S', A <- A'
                state = next_state

                # Add reward to episode reward
                episode_reward += reward

                # If S is terminal, then episode is done (exit loop)
                done = terminated or truncated

            # Add episode reward to list
            self.episode_rewards.append(episode_reward)


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
    rl_loop = QLearning(env)
    rl_loop.learn(num_episodes=train_episodes)

    # Plot the results
    plot_episode_rewards(rl_loop.episode_rewards, rl_loop.name)


if __name__ == "__main__":
    run()
