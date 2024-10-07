from rl.algorithms.common.base_agent import BaseAgent
from rl.common.q_value_table import QValueTable
from rl.common.results_logger import ResultsLogger
import numpy as np
from typing import Tuple


class StateActionStats:
    """
    Implements N(s, a) for Monte Carlo averaging:
        Q(s, a) <- Q(s, a) + 1/N(s, a) * (G - Q(s, a))
    In case of off-policy learning, tracks cumulative sum of importance sampling ratios
        C(s, a) <- C(s, a) + W
    """

    def __init__(self, state_space_shape: Tuple[int, ...], action_space_shape: int):
        self.stats = np.zeros(state_space_shape + (action_space_shape,))

    def get(self, state, action):
        return self.stats[state][action]

    def update(self, state, action):
        self.stats[state][action] += 1

    def update_importance_sampling(self, state, action, importance_sampling_ratio):
        # HOMEWORK: Update the cumulative sum of importance sampling ratios
        # C(s, a) <- C(s, a) + W
        self.stats[state][action] += importance_sampling_ratio


class MonteCarloAgent(BaseAgent):
    """
    Superclass for all Monte Carlo agents. Contains common methods and attributes for all Monte Carlo agents.
    """
    def __init__(self, env, gamma, epsilon=None, logger=None, random_seed=None):
        super().__init__(env, gamma, random_seed)
        self.name = "Base Monte Carlo Agent"
        self.epsilon = epsilon

        self.logger = logger if logger else ResultsLogger()

        # Initialise common Monte Carlo agent attributes
        self.q_values = None
        self.policy = None

        self.returns = None
        self.state_action_stats = None

        # TODO: shift to an environment-specific method?
        # Get env shape
        self.state_shape = ()
        for space in self.env.observation_space:
            self.state_shape += (space.n,)

        self.reset()

    @staticmethod
    def _is_subelement_present(subelement, my_list):
        """
        Helps check if a subelement is present in a list of tuples.
        Used to check if state has already been seen in first-visit MC algorithms.

        Simple example:
        _is_subelement_present((1, 2), [(1, 2, 3), (4, 5, 6)])
            True
        """
        for tpl in my_list:
            if subelement == tpl[:len(subelement)]:
                return True
        return False

    def _generate_episode(self, exploring_starts=True):

        episode = []

        # Reset environment
        if exploring_starts:
            state, info = self.env.reset()  # S_0
            action = np.random.randint(0, self.env.action_space.n)  # A_0: choice of {0, 1}
        else:
            state, info = self.env.reset()
            action = self.act(state)

        # Generate an episode
        while True:

            # HOMEWORK: Make a step of the environment (c.f. Gymnasium API: https://gymnasium.farama.org/api/env/)
            next_state, reward, terminated, truncated, info = self.env.step(action)

            # HOMEWORK: Append the state, action, and reward to the episode
            episode.append((state, action, reward))

            # HOMEWORK: establish if done (this is when either of the boolean flags terminated or truncated are True)
            done = terminated or truncated

            if done:
                break

            # HOMEWORK: the next state becomes the current state
            state = next_state

            # HOMEWORK: the next action is selected by the agent
            action = self.act(state)

        # Log timestep
        self.logger.log_timestep(reward)

        return episode

    def reset(self):

        self.q_values = QValueTable(self.state_shape, self.env.action_space.n)
        self.state_action_stats = StateActionStats(self.state_shape, self.env.action_space.n)
        self.logger.reset()
