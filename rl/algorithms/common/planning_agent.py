from td_agent import TemporalDifferenceAgent


class PlanningAgent(TemporalDifferenceAgent):
    """
    Superclass for all Planning agents. Contains common methods and attributes for all Planning agents.
    """
    def __init__(self, env, gamma, alpha, epsilon, n_planning_steps):
        super().__init__(env, gamma, alpha, epsilon)
        self.name = "Base Planning Agent"
        self.n_planning_steps = n_planning_steps

        # Initialise common Planning agent attributes
        self.model = None
        self.cumulative_reward = None
        self.episode_steps = None
        self.episode_rewards = None
        self.reset()

    def reset(self):
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError

    def learn(self, num_episodes=500):
        raise NotImplementedError
