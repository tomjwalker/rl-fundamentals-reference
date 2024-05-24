import numpy as np
from matplotlib import pyplot as plt


# TODO: random seed not reproducible for TD agents at the moment


class Trial:
    def __init__(self, agent_class, environment, sessions=10, episodes_per_session=1000, random_seeds=None,
                 render=False):

        self.random_seeds = random_seeds if random_seeds is not None else [None] * sessions

        self.agent_class = agent_class
        self.agent = None
        self.environment = environment
        self.sessions = sessions
        self.episodes_per_session = episodes_per_session
        self.render = render
        self.all_rewards = []
        self.loggers = []

    def run(self, verbose=True):
        for session in range(self.sessions):
            if verbose:
                print(f"Running session {session + 1}/{self.sessions} for {self.agent_class.__name__}...")
            self.agent = self.agent_class(self.environment, random_seed=self.random_seeds[session])
            self.agent.learn(self.episodes_per_session)
            self.all_rewards.append(self.agent.logger.total_rewards_per_episode)
            self.loggers.append(self.agent.logger)

    def plot(self, color="blue", ax=None, show_std=True):
        """Plots learning curves (rewards per episode). Line is the mean, shaded region is the standard deviation."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        all_rewards = np.array(self.all_rewards)
        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)

        ax.plot(mean_rewards, color=color, label=self.agent_class.__name__)
        if show_std:
            ax.fill_between(range(len(mean_rewards)), mean_rewards - std_rewards, mean_rewards + std_rewards,
                            color=color, alpha=0.3)

        ax.set_xlabel("Episodes")
        ax.set_ylabel("Total reward during episode")
