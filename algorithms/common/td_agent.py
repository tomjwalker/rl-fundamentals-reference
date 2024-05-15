from base_agent import BaseAgent


class TemporalDifferenceAgent(BaseAgent):
    """
    Superclass for all Temporal Difference agents. Contains common methods and attributes for all Temporal Difference
    agents.
    """
    def __init__(self, env, gamma, alpha, epsilon):
        super().__init__(env, gamma)
        self.name = "Base Temporal Difference Agent"
        self.alpha = alpha
        self.epsilon = epsilon

        # Initialise common Temporal Difference agent attributes
        self.q_values = None
        self.policy = None
        self.episode_rewards = None
        self.reset()

    def reset(self):
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError

    def train(self, num_episodes=500):
        # TODO: check default value of num_episodes
        raise NotImplementedError
