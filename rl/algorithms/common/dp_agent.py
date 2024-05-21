from base_agent import BaseAgent


class DynamicProgrammingAgent(BaseAgent):
    """
    Superclass for all Dynamic Programming agents. Contains common methods and attributes for all Dynamic Programming
    agents.
    """
    def __init__(self, env, gamma):
        super().__init__(env, gamma)
        self.name = "Base Dynamic Programming Agent"

        # Initialise common Dynamic Programming agent attributes
        self.state_values = None
        self.policy = None
        self.reset()

    def reset(self):
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError
