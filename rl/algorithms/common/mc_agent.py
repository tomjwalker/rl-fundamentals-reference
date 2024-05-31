from rl.algorithms.common.base_agent import BaseAgent
from rl.common.q_value_table import QValueTable
from rl.common.results_logger import ResultsLogger
import numpy as np
from typing import Tuple


class StateActionCounts:
    """
    Implements N(s, a) for Monte Carlo averaging:
        Q(s, a) <- Q(s, a) + 1/N(s, a) * (G - Q(s, a))
    """

    def __init__(self, state_space_shape: Tuple[int, ...], action_space_shape: int):
        self.values = np.zeros(state_space_shape + (action_space_shape,))

    def get(self, state, action):
        return self.values[state][action]

    def update(self, state, action):
        self.values[state][action] += 1


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
        self.state_action_counts = None

        # TODO: shift to an environment-specific method?
        # Get env shape
        self.state_shape = ()
        for space in self.env.observation_space:
            self.state_shape += (space.n,)

        self.reset()

    def _generate_episode(self, exploring_starts=True):

        episode = []

        if exploring_starts:
            state, info = self.env.reset()  # S_0
            action = np.random.randint(0, self.env.action_space.n)  # A_0: choice of {0, 1}
        else:
            state, info = self.env.reset()
            action = self.act(state)

        # Generate an episode
        while True:
            next_state, reward, terminated, truncated, info = self.env.step(action)
            episode.append((state, action, reward))
            done = terminated or truncated
            if done:
                break
            state = next_state
            action = self.act(state)

        # Log timestep
        self.logger.log_timestep(reward)

        return episode

    def reset(self):

        self.q_values = QValueTable(self.state_shape, self.env.action_space.n)
        self.state_action_counts = StateActionCounts(self.state_shape, self.env.action_space.n)
        self.logger.reset()

    def act(self, state):
        raise NotImplementedError

    def learn(self, num_episodes=500):
        # TODO: check default value of num_episodes
        raise NotImplementedError
