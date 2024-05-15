class BaseAgent:
    """
    Base class from which all agents inherit. Contains common methods and attributes for all the following agents:
    - Dynamic programming // Policy iteration
    - Dynamic programming // Value iteration
    - Monte Carlo // On-policy
    - Monte Carlo // Off-policy
    - Temporal difference // SARSA
    - Temporal difference // Q-learning
    - Temporal difference // Expected SARSA
    - Planning // Dyna
    - Planning // Dyna-Q+
    """
    def __init__(self, env, gamma):

        self.name = "Base Agent"
        self.env = env
        self.gamma = gamma

        self.policy = None
        self.reset()

    def reset(self):
        """
        Separating out a reset method from the __init__ method allows for the agent to be reset without having to be
        re-instantiated. Can help for reusing the same agent across multiple experiments, clarity, and efficiency.
        """
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError

    def train(self, num_episodes=500):
        raise NotImplementedError
