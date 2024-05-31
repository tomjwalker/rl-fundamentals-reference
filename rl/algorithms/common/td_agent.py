from rl.algorithms.common.base_agent import BaseAgent
from rl.common.policy import EpsilonGreedyPolicy
from rl.common.q_value_table import QValueTable
from rl.common.results_logger import ResultsLogger
import numpy as np


class TemporalDifferenceAgent(BaseAgent):
    """
    Superclass for all Temporal Difference agents. Contains common methods and attributes for all Temporal Difference
    agents.
    """
    def __init__(self, env, gamma, alpha, epsilon, logger=None, random_seed=None):
        super().__init__(env, gamma, random_seed)
        self.name = "Base Temporal Difference Agent"
        self.alpha = alpha
        self.epsilon = epsilon

        self.logger = logger if logger else ResultsLogger()

        # Initialise common Temporal Difference agent attributes
        self.q_values = None
        self.policy = None
        self.reset()

    def reset(self):
        self.q_values = QValueTable((self.env.observation_space.n,), self.env.action_space.n)
        self.policy = EpsilonGreedyPolicy(self.epsilon, self.env.action_space.n)
        self.logger.reset()

    def act(self, state):
        return self.policy.select_action(state, self.q_values)

    def learn(self, num_episodes=500):
        # TODO: check default value of num_episodes
        raise NotImplementedError
