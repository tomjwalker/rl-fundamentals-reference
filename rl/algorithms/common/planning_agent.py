from rl.algorithms.common.td_agent import TemporalDifferenceAgent


class PlanningAgent(TemporalDifferenceAgent):
    """Superclass for legacy planning agents."""

    def __init__(self, env, gamma, alpha, epsilon, n_planning_steps, logger=None, random_seed=None):
        super().__init__(env, gamma, alpha, epsilon, logger, random_seed)
        self.name = "Base Planning Agent"
        self.n_planning_steps = n_planning_steps

        self.model = None
        self.reset()

    def reset(self):
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError

    def learn(self, num_episodes=500):
        raise NotImplementedError
