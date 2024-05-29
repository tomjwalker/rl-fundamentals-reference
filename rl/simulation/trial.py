import numpy as np
from matplotlib import pyplot as plt


# TODO: random seed not reproducible for TD agents at the moment


class Trial:
    def __init__(self, agent_class, environment_class=None, environment=None, sessions=10, episodes_per_session=1000,
                 random_seeds=None,
                 render=False, **agent_kwargs):

        self.random_seeds = random_seeds if random_seeds is not None else [None] * sessions

        self.agent_class = agent_class
        self.agent = None
        self.environment_class = environment_class
        self.environment = environment
        self.sessions = sessions
        self.episodes_per_session = episodes_per_session
        self.render = render
        self.agent_kwargs = agent_kwargs
        self.all_rewards = []
        self.loggers = []

    def run(self, verbose=True):
        for session in range(self.sessions):

            if verbose:
                print(f"Running session {session + 1}/{self.sessions} for {self.agent_class.__name__}...")

            # Instantiate env. Different modes depending on whether `environment_class` or `environment` is provided
            if self.environment_class is not None:
                # Re-instantiating helpful for the case where environment is non-stationary
                self.environment = self.environment_class()
            elif self.environment is not None:
                # Passing an instantiated environment helpful for Gym environments with gym.make()
                self.environment.reset()
            else:
                raise ValueError("Either `environment_class` or `environment` must be provided.")

            # Update random_seed for the current session
            self.agent_kwargs["random_seed"] = self.random_seeds[session]

            self.agent = self.agent_class(self.environment, **self.agent_kwargs)
            self.agent.learn(self.episodes_per_session)
            self.all_rewards.append(self.agent.logger.total_rewards_per_episode)
            self.loggers.append(self.agent.logger)

    def plot(self, series_type, color="blue", ax=None, show_std=True, std_alpha=0.3):
        """Plots learning curves (rewards per episode). Line is the mean, shaded region is the standard deviation."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        if series_type == "total_rewards_per_episode":
            series = [logger.total_rewards_per_episode for logger in self.loggers]
            xlabel = "Episodes"
            ylabel = "Sum of rewards during episode"
        elif series_type == "steps_per_episode":
            series = [logger.steps_per_episode for logger in self.loggers]
            xlabel = "Episodes"
            ylabel = "Steps per episode"
        elif series_type == "cumulative_rewards":
            series = [logger.cumulative_rewards for logger in self.loggers]
            xlabel = "Timesteps"
            ylabel = "Cumulative reward"
        else:
            raise ValueError(f"Unknown series type: {series_type}")

        # Find the minimum length of the series
        min_length = min([len(s) for s in series])

        # Trim the series to the minimum length
        trimmed_series = np.array([s[:min_length] for s in series])

        mean_series = np.mean(trimmed_series, axis=0)
        std_series = np.std(trimmed_series, axis=0)

        ax.plot(mean_series, color=color, label=self.agent_class.__name__)
        if show_std:
            ax.fill_between(range(len(mean_series)), mean_series - std_series, mean_series + std_series,
                            color=color, alpha=std_alpha)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
