from rl.algorithms.common.td_agent import TemporalDifferenceAgent
import numpy as np
from collections import defaultdict    # For model, with default value of empty list
import random    # For random choice of state and action from model during planning


class DynaModel:
    """
    Model for a Dyna agent, storing the (state, action) -> (reward, next_state) mapping.

    Model is a nested dictionary: first key is state, second key is action, value is (reward, next_state).
    For the purposes of this repo, we assume a deterministic environment, so the mapping is unique.
    """

    def __init__(self, random_seed=None):
        self.model = defaultdict(dict)
        self.random_seed = random_seed
        if random_seed:
            self._set_random_seed()

    def _set_random_seed(self):
        random.seed(self.random_seed)

    def add(self, state, action, reward, next_state):
        self.model[state][action] = (reward, next_state)

    def get(self, state, action):
        return self.model[state][action]

    def sample_state_action(self):
        state = random.choice(list(self.model.keys()))
        action = random.choice(list(self.model[state].keys()))
        return state, action


class Dyna(TemporalDifferenceAgent):

    def __init__(self, env, alpha=0.5, gamma=1.0, epsilon=0.1, n_planning_steps=5, logger=None, random_seed=None):

        # Initialise common Temporal Difference agent attributes: q_values, policy, episode_rewards
        super().__init__(env, gamma, alpha, epsilon, logger, random_seed)

        # Initialise Dyna-specific attributes
        self.name = "Dyna"
        self.n_planning_steps = n_planning_steps

        # Initialise attributes which reset on each new episode
        self.model = None
        self.reset()

    def reset(self):

        # Reset common Temporal Difference agent attributes: q_values, policy, episode_rewards
        super().reset()

        # Initialise Dyna-specific attributes
        self.model = DynaModel(self.random_seed)

    def learn(self, num_episodes=500):

        for episode in range(num_episodes):

            # Initialise S (**a**)
            state, _ = self.env.reset()

            # Loop over each step of episode, until S is terminal
            done = False
            while not done:

                # Choose A from S using policy derived from Q (epsilon-greedy) (**b**)
                action = self.act(state)

                # Take action A, observe R, S' (**c**)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Update Q(S, A) (**d**)
                td_target = reward + self.gamma * self.q_values.get(next_state, self.q_values.get_max_action(next_state))
                td_error = td_target - self.q_values.get(state, action)
                new_value = self.q_values.get(state, action) + self.alpha * td_error
                self.q_values.update(state, action, new_value)

                # Update logs
                self.logger.log_timestep(reward)

                # Update model (**e**).
                self.model.add(state, action, reward, next_state)

                # Loop for n planning steps, and perform planning updates (**f**)
                for _ in range(self.n_planning_steps):

                    # Choose a random, previously observed state and action (**f.i, f.ii**)
                    state, action = self.model.sample_state_action()

                    # `next_state_plan` ensures next state from learning is not mixed up with next state from
                    # planning (for S <- S')
                    (reward, next_state_plan) = self.model.get(state, action)

                    # Update Q(S, A), taking as target the q-learning TD target (R + gamma * max_a Q(S', a)) (**f.iv**)
                    td_target = reward + self.gamma * np.max(self.q_values.get(next_state_plan))
                    td_error = td_target - self.q_values.get(state, action)
                    new_value = self.q_values.get(state, action) + self.alpha * td_error
                    self.q_values.update(state, action, new_value)

                # S <- S'
                state = next_state

                # If S is terminal, then episode is done (exit loop)
                done = terminated or truncated

            # Update logs
            self.logger.log_episode()
