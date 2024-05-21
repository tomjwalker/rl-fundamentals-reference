from base_agent import BaseAgent


class MonteCarloAgent(BaseAgent):
    """
    Superclass for all Monte Carlo agents. Contains common methods and attributes for all Monte Carlo agents.
    """
    def __init__(self, env, gamma, epsilon=None):
        super().__init__(env, gamma)
        self.name = "Base Monte Carlo Agent"
        self.epsilon = epsilon

        # Initialise common Monte Carlo agent attributes
        self.q_values = None
        self.policy = None
        self.returns = None
        self.reset()

    def reset(self):
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError

    def learn(self, num_episodes=500):
        # TODO: check default value of num_episodes
        raise NotImplementedError
