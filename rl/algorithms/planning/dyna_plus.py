from rl.algorithms.planning.dyna import Dyna, DynaModel    # DynaPlus inherits from Dyna, so we can reuse a lot of the
from rl.common.q_value_table import QValueTable

import numpy as np


# TODO: cleaner way to deal with observation spaces of different shapes (see QValueTable and TimeSinceLastEncountered)

class DynaPlusModel(DynaModel):
    """
    Model object for the Dyna-plus algorithm, storing the (state, action) -> (reward, next_state) mapping.

    Additional to the DynaModel, when this class initialises a new state, it will include transitions for all
    actions, including those not yet taken (in these instances, the modelled reward is 0 and the next state is the
    current state).
    """

    def __init__(self, num_actions, random_seed=None):
        super().__init__(random_seed)
        self.num_actions = num_actions

    def add(self, state, action, reward, next_state):

        # If state is newly encountered, initialise all actions with (reward=0, next_state=state)
        if state not in self.model.keys():
            for a in range(self.num_actions):
                self.model[state][a] = (0, state)

        # Add the actual transition
        self.model[state][action] = (reward, next_state)


class TimeSinceLastEncountered(QValueTable):
    """
    Implements the tau(s, a) table for Dyna+ algorithm.

    This is a NumPy array, same size as the Q-value table, initialised with zeros, so can base this class on the
    QValueTable class, which comes with methods:
    - get(state, action) -> value
    - update(state, action, value)

    We can extend this class with a method to increment all values by 1, except for a single (state, action) pair, which
    is reset to 0.
    """

    def __init__(self, num_states, num_actions):
        super().__init__(num_states, num_actions)

    def increment(self, state, action):
        self.values += 1
        self.update(state, action, 0)


class DynaPlus(Dyna):

    def __init__(self, env, alpha=0.5, gamma=1.0, epsilon=0.1, n_planning_steps=5, kappa=0.001,
                 logger=None, random_seed=None):

        # Initialise attributes common to Dyna
        super().__init__(env, alpha, gamma, epsilon, n_planning_steps, logger, random_seed)

        self.name = "Dyna+"
        self.kappa = kappa
        # TODO: ? N.B. this is initialised outside the reset() method, as the total number of steps taken is not reset
        #  when the environment is reset at the end of an episode
        self.model = None
        self.time_since_last_encountered = None
        self.reset()

    def reset(self):
        super().reset()

        self.model = DynaPlusModel(self.env.action_space.n, self.random_seed)
        self.time_since_last_encountered = TimeSinceLastEncountered((self.env.observation_space.n,),
                                                                    self.env.action_space.n)

    def learn(self, num_episodes=500):

        for episode in range(num_episodes):

            # Initialise S (**a**)
            state, _ = self.env.reset()

            # Initialise logging variables
            episode_reward = 0
            episode_steps = 0

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

                # Update time since last encountered
                self.time_since_last_encountered.increment(state, action)

                # Loop for n planning steps, and perform planning updates (**f**)
                for _ in range(self.n_planning_steps):

                    # Choose a random, previously observed state and any action (**f.i, f.ii**)
                    (state, action) = self.model.sample_state_action()

                    # TODO: different from Dyna (no values as lists)
                    # Get reward and next state from model (**f.iii**)
                    # `next_state_plan` ensures next state from learning is not mixed up with next state from planning (for S <- S')
                    reward, next_state_plan = self.model.get(state, action)

                    # Get time since last encountered for (s, a)
                    time_since_last_encountered = self.time_since_last_encountered.get(state, action)

                    # Update Q(S, A), taking as target the q-learning TD target **with Dyna-Q+** additional term:
                    # TD_target = R + gamma * max_a Q(S', a) + kappa * sqrt(time_since_last_encountered) (**f.iv**)
                    reward_with_bonus = reward + self.kappa * np.sqrt(time_since_last_encountered)
                    td_target = reward_with_bonus + self.gamma * np.max(self.q_values.get(next_state_plan))
                    td_error = td_target - self.q_values.get(state, action)
                    new_value = self.q_values.get(state, action) + self.alpha * td_error
                    self.q_values.update(state, action, new_value)

                # S <- S'
                state = next_state

                # If S is terminal, then episode is done (exit loop)
                done = terminated or truncated

            # Update logs
            self.logger.log_episode()
