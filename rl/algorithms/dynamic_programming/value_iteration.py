import numpy as np
from typing import Tuple
from rl.environment.dynamic_programming.jacks_car_rental import JacksCarRental
from rl.utils.general import set_filepath
import os
from rl.algorithms.dynamic_programming.viz import plot_policy_and_value


class ValueIteration:
    def __init__(self, env: JacksCarRental, gamma: float = 0.9, theta: float = 1e-8) -> None:
        """
        Initialise the ValueIteration algorithm with the given environment.

        Args:
            env (JacksCarRental): The environment to solve.
            gamma (float): The discount factor.
            theta (float): A small threshold for determining convergence.
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta

        # TODO: might need to return to this if env refactored following Gymnasium API
        self.max_cars: int = env.max_cars
        self.policy: np.ndarray = np.zeros(
            (self.max_cars + 1, self.max_cars + 1), dtype=np.int8
        )  # int8 fits action range
        self.value: np.ndarray = np.zeros(
            (self.max_cars + 1, self.max_cars + 1), dtype=np.float32
        )

    def save_artefacts(self, save_name: str) -> None:
        """
        Save the current policy and value function to disk.

        Args:
            save_name (str): The name to use when saving the policy and value arrays.
        """
        policy_dir = "./.data/dynamic_programming/policy_iteration/policy"
        value_dir = "./.data/dynamic_programming/policy_iteration/value"

        # Ensure directories exist
        os.makedirs(policy_dir, exist_ok=True)
        os.makedirs(value_dir, exist_ok=True)

        policy_filepath = policy_dir + "/" + save_name + ".npy"
        value_filepath = value_dir + "/" + save_name + ".npy"
        np.save(set_filepath(policy_filepath), self.policy)
        np.save(set_filepath(value_filepath), self.value)

    def _update_expected_return_array(self) -> None:
        """
        Helper function for calculating expected returns efficiently.

        This updates the `gamma P^{(1)T} V(s') P^{(2)}` matrix for all states s' in S.

        See lecture notes for further details.
        """
        self.expected_value_matrix = self.env.get_expected_value(self.value, self.gamma)

    def _get_expected_return(
        self, state_1_next_morning: int, state_2_next_morning: int, action: int
    ) -> float:
        """
        Calculates the expected return for a given state and action efficiently, using stored matrices.

        This evaluates:

            sum_{r,s'} p(s', r | s, a) [r + gamma v(s')] =
                EXPECTED_VALUE_MATRIX(s_1^dagger, s_2^dagger) - R_a * |a|

        Args:
            state_1_next_morning (int): Number of cars at location 1 the next morning.
            state_2_next_morning (int): Number of cars at location 2 the next morning.
            action (int): The action taken (number of cars moved from location 1 to 2).

        Returns:
            float: The expected return for the given state and action.
        """
        return (
            self.expected_value_matrix[state_1_next_morning, state_2_next_morning]
            - self.env.move_cost * np.abs(action)
        )

    def value_iteration(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the value iteration algorithm.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The optimal policy and value function.
        """
        loop_idx: int = 0

        # Lecture algorithm pseudo-code: "repeat:"
        while True:

            # HOMEWORK: delta <- 0
            delta: float = 0

            # Efficiently calculate expected returns for all states
            self._update_expected_return_array()

            # Initialise available actions
            available_actions = np.arange(-self.env.max_move_cars, self.env.max_move_cars + 1)

            # For each s in S
            for state_1 in range(self.max_cars + 1):
                for state_2 in range(self.max_cars + 1):

                    # Initialise best_action and best_return for this state
                    # A conditional if block will then enact the argmax_a{expected_return} operation
                    best_action = None
                    best_return = -np.inf

                    # HOMEWORK: store old value ("v <- V(s)"). c.f. self.value
                    old_value: float = self.value[state_1, state_2]

                    # Loop through all possible actions
                    for action in available_actions:

                        # HOMEWORK: the environment object has a method that computes the next state given the current state
                        # and the action.
                        # Use this method to compute the next state: next_state = env.compute_next_state(...)
                        next_state = self.env.compute_next_state((state_1, state_2), action)

                        # If these new states fall outside the range of possible states, then continue
                        if next_state is None:
                            continue

                        # Unpack the next state
                        next_state_1, next_state_2 = next_state

                        # expected return = sum_{s', r} p(s', r|s, a) [r + gamma V(s')]
                        # Use helper function _get_expected_return for this calculation
                        expected_return: float = self._get_expected_return(
                            next_state_1, next_state_2, action
                        )

                        # Want max value (max_a(expected_return) for this state)
                        # Also want to store the action that led to this max value
                        if expected_return > best_return:
                            best_return = expected_return
                            best_action = action

                    # HOMEWORK: update value as best return ("V_*(s) <- max_a{expected_return}")
                    self.value[state_1, state_2] = best_return

                    # HOMEWORK: Update policy with best action
                    self.policy[state_1, state_2] = best_action

                    # HOMEWORK: delta <- max(delta, |v - V(s)|)
                    delta = max(delta, np.abs(old_value - self.value[state_1, state_2]))

            print(f"Value improvement: loop {loop_idx}, delta = {delta}")

            # Save the policy and value at the end of each loop
            self.save_artefacts(f"value_iteration_{loop_idx}")

            loop_idx += 1

            # HOMEWORK START: (2 lines)
            # If delta < self.theta, then the value function has converged, and policy evaluation can stop (break loop)
            if delta < self.theta:
                break
            # HOMEWORK END

        return self.policy, self.value


if __name__ == "__main__":
    env = JacksCarRental()
    value_iteration = ValueIteration(env)
    policy, value = value_iteration.value_iteration()

    # Plot the policy and value
    plot_policy_and_value(policy, value)
