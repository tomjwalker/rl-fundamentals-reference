from rl.algorithms.planning.dyna import Dyna, DynaModel  # DynaPlus inherits from Dyna, so we can reuse a lot of the
from rl.common.q_value_table import QValueTable

import numpy as np
from typing import Optional, Tuple, Union
from rl.common.results_logger import ResultsLogger

# TODO: cleaner way to deal with observation spaces of different shapes (see QValueTable and TimeSinceLastEncountered)


class DynaPlusModel(DynaModel):
    """
    Model object for the Dyna-plus algorithm, storing the (state, action) -> (reward, next_state) mapping.

    Additional to the DynaModel, when this class initialises a new state, it will include transitions for all
    actions, including those not yet taken (in these instances, the modelled reward is 0 and the next state is the
    current state).

    Attributes:
        num_actions (int): The number of actions available in the environment.
    """

    def __init__(self, num_actions: int, random_seed: Optional[int] = None) -> None:
        """
        Initialise the DynaPlusModel.

        Args:
            num_actions (int): Number of actions available in the environment.
            random_seed (Optional[int]): Random seed for reproducibility.
        """
        super().__init__(random_seed)
        self.num_actions = num_actions

    def add(
            self, state: Union[int, Tuple[int, ...]], action: int,
            reward: float, next_state: Union[int, Tuple[int, ...]]
    ) -> None:
        """
        Add a transition to the model.

        Args:
            state (Union[int, Tuple[int, ...]]): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (Union[int, Tuple[int, ...]]): The next state resulting from the action.
        """
        if state not in self.model.keys():
            for a in range(self.num_actions):
                # HOMEWORK: If state is newly encountered, initialise all actions with (reward=0, next_state=state)
                self.model[state][a] = (0.0, state)

        # HOMEWORK: Add the actual transition
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

    Attributes:
        values (np.ndarray): The table representing the time since each (state, action) pair was last encountered.
    """

    def __init__(self, num_states: int, num_actions: int) -> None:
        """
        Initialise the TimeSinceLastEncountered table.

        Args:
            num_states (int): The number of states in the environment.
            num_actions (int): The number of actions available in the environment.
        """
        super().__init__((num_states,), num_actions)

    def increment(self, state: Union[int, Tuple[int, ...]], action: int) -> None:
        """
        Increment all values in the table by 1, except for the specified (state, action) pair which is reset to 0.

        Args:
            state (Union[int, Tuple[int, ...]]): The state to reset.
            action (int): The action to reset.
        """
        # HOMEWORK: Increment values for all (S, A) pairs by 1
        self.values += 1

        # HOMEWORK: Except for the (state, action) pair, which is reset to 0 (use self.update with the right arguments)
        self.update(state, action, 0)


class DynaPlus(Dyna):
    """
    Dyna-Q+ algorithm implementation, extending the Dyna algorithm by incorporating exploration bonuses.

    Attributes:
        kappa (float): Exploration bonus coefficient.
        model (DynaPlusModel): The model used to simulate experience for planning.
        time_since_last_encountered (TimeSinceLastEncountered): Table to track the time since each (state, action) pair
            was last encountered.
    """

    def __init__(self, env, alpha: float = 0.5, gamma: float = 1.0, epsilon: float = 0.1, n_planning_steps: int = 5,
                 kappa: float = 0.001, logger: Optional[ResultsLogger] = None, random_seed: Optional[int] = None
                 ) -> None:
        """
        Initialise the DynaPlus agent.

        Args:
            env: The environment to interact with.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Probability of choosing a random action (epsilon-greedy policy).
            n_planning_steps (int): Number of planning steps per real step.
            kappa (float): Exploration bonus coefficient.
            logger (Optional[ResultsLogger]): Logger for recording training progress.
            random_seed (Optional[int]): Random seed for reproducibility.
        """
        # Initialise attributes common to Dyna
        super().__init__(env, alpha, gamma, epsilon, n_planning_steps, logger, random_seed)

        self.name = "Dyna+"
        self.kappa = kappa
        # TODO: ? N.B. this is initialised outside the reset() method, as the total number of steps taken is not reset
        #  when the environment is reset at the end of an episode
        self.model: Optional[DynaPlusModel] = None
        self.time_since_last_encountered: Optional[TimeSinceLastEncountered] = None
        self.reset()

    def reset(self) -> None:
        """
        Reset the agent for a new episode.
        """
        super().reset()

        self.model = DynaPlusModel(self.env.action_space.n, self.random_seed)
        self.time_since_last_encountered = TimeSinceLastEncountered(
            self.env.observation_space.n,
            self.env.action_space.n
        )

    def learn(self, num_episodes: int = 500) -> None:
        """
        Train the agent using the Dyna-Q+ algorithm.

        Args:
            num_episodes (int): Number of episodes to train the agent for.
        """
        # HOMEWORK STARTS: Implement the Dyna-Q+ algorithm (~25-30 lines).

        for episode in range(num_episodes):
            # Initialise S (**a**)
            state: Union[int, Tuple[int, ...]]
            state, _ = self.env.reset()

            # Loop over each step of episode, until S is terminal
            done: bool = False
            while not done:
                # Choose A from S using policy derived from Q (epsilon-greedy) (**b**)
                action: int = self.act(state)

                # Take action A, observe R, S' (**c**)
                next_state: Union[int, Tuple[int, ...]]
                reward: float
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Update Q(S, A) (**d**)
                td_target: float = reward + self.gamma * self.q_values.get(next_state, self.q_values.get_max_action(next_state))  # NoQA
                td_error: float = td_target - self.q_values.get(state, action)
                new_value: float = self.q_values.get(state, action) + self.alpha * td_error
                self.q_values.update(state, action, new_value)

                # Update logs
                if self.logger:
                    self.logger.log_timestep(reward)

                # Update model (**e**).
                self.model.add(state, action, reward, next_state)

                # Update time since last encountered
                self.time_since_last_encountered.increment(state, action)

                # Loop for n planning steps, and perform planning updates (**f**)
                for _ in range(self.n_planning_steps):
                    # Choose a random, previously observed state and any action (**f.i, f.ii**)
                    state_plan: Union[int, Tuple[int, ...]]
                    action_plan: int
                    state_plan, action_plan = self.model.sample_state_action()

                    # TODO: different from Dyna (no values as lists)
                    # Get reward and next state from model (**f.iii**)
                    # `next_state_plan` ensures next state from learning is not mixed up with next state from planning
                    # (for S <- S')
                    reward_plan: float
                    next_state_plan: Union[int, Tuple[int, ...]]
                    reward_plan, next_state_plan = self.model.get(state_plan, action_plan)

                    # Get time since last encountered for (s, a)
                    time_since_last_encountered: int = self.time_since_last_encountered.get(state_plan, action_plan)

                    # Update Q(S, A), taking as target the q-learning TD target **with Dyna-Q+** additional term:
                    # TD_target = R + gamma * max_a Q(S', a) + kappa * sqrt(time_since_last_encountered) (**f.iv**)
                    reward_with_bonus: float = reward_plan + self.kappa * np.sqrt(time_since_last_encountered)
                    td_target = reward_with_bonus + self.gamma * np.max(self.q_values.get(next_state_plan))
                    td_error = td_target - self.q_values.get(state_plan, action_plan)
                    new_value = self.q_values.get(state_plan, action_plan) + self.alpha * td_error
                    self.q_values.update(state_plan, action_plan, new_value)

                # S <- S'
                state = next_state

                # If S is terminal, then episode is done (exit loop)
                done = terminated or truncated

            # Update logs
            if self.logger:
                self.logger.log_episode()

        # HOMEWORK ENDS
