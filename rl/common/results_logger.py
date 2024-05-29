import numpy as np


class ResultsLogger:
    def __init__(self, log_total_reward=True, log_steps_per_episode=True, log_cumulative_reward=True):

        # Boolean flags for selecting which logging to perform
        self.log_total_reward = log_total_reward
        self.log_steps_per_episode = log_steps_per_episode
        self.log_cumulative_reward = log_cumulative_reward

        # TODO: rename?
        # Intra-episode logging
        self.total_rewards_per_episode = []
        self.steps_per_episode = []
        self.episode_reward = 0

        # TODO: rename?
        # Inter-episode logging
        self.cumulative_rewards = []
        self.total_rewards = 0
        self.steps = 0

    def log_timestep(self, reward):
        if self.log_cumulative_reward:
            if self.cumulative_rewards:
                # List already populated
                self.cumulative_rewards.append(self.cumulative_rewards[-1] + reward)
            else:
                # Empty list (first reward)
                self.cumulative_rewards.append(reward)
        self.episode_reward += reward
        self.steps += 1

    def log_episode(self):
        """Updates inter-episode logging variables and resets intra-episode logging variables."""
        if self.log_total_reward:
            self.total_rewards_per_episode.append(self.episode_reward)
            self.total_rewards += self.episode_reward
        if self.log_steps_per_episode:
            self.steps_per_episode.append(self.steps)
        self.episode_reward = 0
        self.steps = 0

    def reset(self):
        self.total_rewards_per_episode = []
        self.steps_per_episode = []
        self.cumulative_rewards = []
        self.total_rewards = 0
        self.episode_reward = 0
        self.steps = 0

    def get_stats(self):
        stats = {}
        if self.log_total_reward:
            stats['total_rewards_per_episode'] = self.total_rewards_per_episode
            stats['total_rewards'] = self.total_rewards
            stats['average_reward'] = np.mean(self.total_rewards_per_episode) if self.total_rewards_per_episode else 0
            stats['reward_variance'] = np.var(self.total_rewards_per_episode) if self.total_rewards_per_episode else 0
        if self.log_steps_per_episode:
            stats['steps_per_episode'] = self.steps_per_episode
            stats['average_steps'] = np.mean(self.steps_per_episode) if self.steps_per_episode else 0
        if self.log_cumulative_reward:
            stats['cumulative_rewards'] = self.cumulative_rewards
        return stats
