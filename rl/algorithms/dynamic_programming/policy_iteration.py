import numpy as np
from typing import Tuple
from rl.environment.dynamic_programming.jacks_car_rental import JacksCarRental
from rl.utils.general import set_filepath
import os
from rl.algorithms.dynamic_programming.viz import plot_policy_and_value


class PolicyIteration:
    def __init__(self, env: JacksCarRental, gamma: float = 0.9, theta: float = 1e-8) -> None:
        """
        Initialise the PolicyIteration algorithm with the given environment.

        Args:
            env (JacksCarRental): The environment to solve.
            gamma (float): The discount factor.
            theta (float): A small threshold for determining convergence in policy evaluation.
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta

        # TODO: might need to return to this if env refactored following Gymnasium API
        self.max_cars: int = env.max_cars
        self.policy: np.ndarray = np.zeros((self.max_cars + 1, self.max_cars + 1), dtype=np.int8)  # int8 fits action range    # noqa
        self.value: np.ndarray = np.zeros((self.max_cars + 1, self.max_cars + 1), dtype=np.float32)

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

        This updates the `gamma P^{(1)T} V(s') P^{(2)}` matrix, for all states s' in S.

        See lecture notes for further details.
        """
        self.expected_value_matrix = self.env.get_expected_value(self.value, self.gamma)

    def _get_expected_return(self, state_1_next_morning: int, state_2_next_morning: int, action: int) -> float:
        """
        Calculates the expected return for a given state and action efficiently, using stored matrices.

        This evaluates:

            sum_{r,s'} p(s', r | s, a) (r + gamma v(s')) =
                EXPECTED_VALUE_MATRIX(s_1^dagger, s_2^dagger) + R_a |a|

        Args:
            state_1_next_morning (int): Number of cars at location 1 the next morning.
            state_2_next_morning (int): Number of cars at location 2 the next morning.
            action (int): The action taken (number of cars moved from location 1 to 2).

        Returns:
            float: The expected return for the given state and action.
        """
        expected_value_next_state = self.expected_value_matrix[state_1_next_morning, state_2_next_morning]
        expected_return = expected_value_next_state - self.env.move_cost * np.abs(action)
        return expected_return

    def policy_evaluation(self) -> None:
        """
        Perform policy evaluation to update the value function using the current policy.
        """
        # Lecture algorithm pseudo-code: "repeat:"
        while True:

            # HOMEWORK: delta <- 0
            delta: float = 0

            # Efficiently calculate expected returns for all states
            self._update_expected_return_array()

            for state_1 in range(self.max_cars + 1):
                for state_2 in range(self.max_cars + 1):

                    # HOMEWORK: store old value ("v <- V(s)"). c.f. self.value
                    old_value: float = self.value[state_1, state_2]

                    # HOMEWORK: retrieve a <- pi(s) as the action to take (deterministic policy). c.f. self.policy
                    action: int = self.policy[state_1, state_2]

                    # HOMEWORK: the environment object has a method that computes the next state given the current state
                    # and the action.
                    # Use this method to compute the next state: next_state = env.compute_next_state(...)
                    next_state = self.env.compute_next_state((state_1, state_2), action)

                    # The next state might be invalid (e.g. if the action is to move more cars than are available).
                    # The environment method returns None in this case, and this control structure skips the rest of the
                    # loop iteration.
                    if next_state is None:
                        continue  # Skip invalid states

                    # Unpack the next state
                    next_state_1, next_state_2 = next_state

                    # HOMEWORK: Use helper function _get_expected_return for this step
                    # This calculates "expected return = sum_{s', r} p(s', r|s, a) [r + gamma V(s')]" efficiently
                    expected_return: float = self._get_expected_return(next_state_1, next_state_2, action)

                    # HOMEWORK: V(s) <- expected return
                    self.value[state_1, state_2] = expected_return

                    # HOMEWORK: delta <- max(delta, |v - V(s)|)
                    delta = max(delta, np.abs(old_value - self.value[state_1, state_2]))

            # HOMEWORK START: (2 lines)
            # If delta < self.theta, then the value function has converged, and policy evaluation can stop (break loop)
            if delta < self.theta:
                break
            # HOMEWORK END

    def policy_improvement(self) -> bool:
        """
        Perform policy improvement to update the policy based on the current value function.

        Returns:
            bool: True if the policy is stable (no changes), False otherwise.
        """
        # HOMEWORK: policy_stable <- True
        policy_stable: bool = True

        # Initialise available actions
        available_actions = np.arange(-self.env.max_move_cars, self.env.max_move_cars + 1)

        # For each s in S
        for state_1 in range(self.max_cars + 1):
            for state_2 in range(self.max_cars + 1):

                # HOMEWORK: store old action ("old_action <- pi(s)")
                old_action: int = self.policy[state_1, state_2]

                # Initialise action_returns for all possible actions
                # Each element in action_returns corresponds to an action in available_actions (running from
                # [-max_move_cars, max_move_cars] (inclusive))
                # This will make the argmax calculation easier (list elements will be replaced by expected rewards for
                # each action)
                action_returns: list = []

                # For each action in A(s)...
                for action in available_actions:

                    # HOMEWORK: the environment object has a method that computes the next state given the current state
                    # and the action.
                    # Use this method to compute the next state: next_state = env.compute_next_state(...)
                    next_state = self.env.compute_next_state((state_1, state_2), action)

                    if next_state is None:
                        action_returns.append(-np.inf)    # This ensures invalid actions are never selected by argmax
                        continue  # Skip invalid actions

                    # Unpack the next state
                    next_state_1, next_state_2 = next_state

                    # HOMEWORK: Use helper function _get_expected_return for this step
                    # This calculates "expected return = sum_{s', r} p(s', r|s, a) [r + gamma V(s')]" efficiently
                    expected_return: float = self._get_expected_return(next_state_1, next_state_2, action)

                    # HOMEWORK: Update action_returns list with expected return for this action
                    action_returns.append(expected_return)

                # HOMEWORK: Once all actions have been evaluated, select the best action
                # (use np.argmax on action_returns to find the index for the best action in available_actions)
                best_action: int = available_actions[np.argmax(action_returns)]

                # HOMEWORK: Update policy with best action
                self.policy[state_1, state_2] = best_action

                # HOMEWORK START: (2 lines)
                # If old_action != pi(s), then policy_stable <- False
                if old_action != self.policy[state_1, state_2]:
                    policy_stable = False
                # HOMEWORK END

        return policy_stable

    def policy_iteration(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the policy iteration algorithm.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The optimal policy and value function.
        """
        loop: int = 0
        while True:

            print(f"Policy evaluation: loop {loop}")

            # HOMEWORK: Perform policy evaluation
            self.policy_evaluation()

            self.save_artefacts(save_name=f"policy_evaluation_{loop}")

            print(f"Policy improvement: loop {loop}")

            # HOMEWORK: Perform policy improvement
            policy_stable: bool = self.policy_improvement()

            self.save_artefacts(save_name=f"policy_improvement_{loop}")

            # HOMEWORK START: (2 lines), if the policy is stable, break the loop (policy iteration is complete)
            if policy_stable:
                break
            # HOMEWORK END

            loop += 1

        return self.policy, self.value


if __name__ == "__main__":
    env = JacksCarRental()
    policy_iteration = PolicyIteration(env)
    policy, value = policy_iteration.policy_iteration()

    # Plot the policy and value
    plot_policy_and_value(policy, value)
