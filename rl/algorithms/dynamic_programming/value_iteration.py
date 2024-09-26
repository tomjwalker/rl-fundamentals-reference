import numpy as np
from rl.environment.dynamic_programming.jacks_car_rental import JacksCarRental
from rl.utils.general import set_filepath
import os
from rl.algorithms.dynamic_programming.viz import plot_policy_and_value


class ValueIteration:
    def __init__(self, env, gamma=0.9, theta=1e-8):
        self.env = env
        self.gamma = gamma
        self.theta = theta

        # TODO: might need to return to this if env refactored following Gymnasium API
        self.max_cars = env.max_cars
        self.policy = np.zeros((self.max_cars + 1, self.max_cars + 1), dtype=np.int8)  # int8 fits action range
        self.value = np.zeros((self.max_cars + 1, self.max_cars + 1), dtype=np.float32)

    def save_artefacts(self, save_name):
        policy_dir = "./.data/dynamic_programming/policy_iteration/policy"
        value_dir = "./.data/dynamic_programming/policy_iteration/value"

        # Ensure directories exist
        os.makedirs(policy_dir, exist_ok=True)
        os.makedirs(value_dir, exist_ok=True)

        policy_filepath = policy_dir + "/" + save_name + ".npy"
        value_filepath = value_dir + "/" + save_name + ".npy"
        np.save(set_filepath(policy_filepath), self.policy)
        np.save(set_filepath(value_filepath), self.value)

    def _update_expected_return_array(self):
        """
        Helper function for calculating expected returns efficiently.

        This updates the `gamma P^{(1)T} V(s') P^{(2)}` matrix, for all states s' in S.

        See lecture notes for further details.
        """
        self.expected_value_matrix = self.env.get_expected_value(self.value, self.gamma)

    def _get_expected_return(self, state_1_next_morning, state_2_next_morning, action):
        """
        Calculates the expected return for a given state and action efficiently, using stored matrices.

        This evaluates:

            sum_{r,s'} p(s', r | s, a) (r + gamma v(s')) =
                EXPECTED_VALUE_MATRIX(s_1^dagger, s_2^dagger) + R_a |a|
        """
        return self.expected_value_matrix[state_1_next_morning, state_2_next_morning] - self.env.move_cost * \
            np.abs(action)

    def value_iteration(self):
        loop_idx = 0  # For logging training progress
        while True:
            delta = 0

            # Efficiently calculate expected returns for all states
            self._update_expected_return_array()

            available_actions = np.arange(-self.env.max_move_cars, self.env.max_move_cars + 1)
            for state_1 in range(self.max_cars + 1):
                for state_2 in range(self.max_cars + 1):

                    # Initialise {best_action, best_return} dict for this state
                    best_action = None
                    best_return = -np.inf

                    # v <- V(s)
                    old_value = self.value[state_1, state_2]

                    # ==================================================================================================
                    # V(s) <- max_a sum_{s', r} p(s', r|s, a) [r + gamma V(s')]
                    # ==================================================================================================

                    # Loop through all possible actions
                    for action in available_actions:

                        # Calculate s'' (the next morning's state after redistribution.
                        # Action a moves cars from location 1 to location 2, so:
                        #    - s''_1 = s_1 - a
                        #    - s''_2 = s_2 + a
                        state_1_morning = state_1 - action
                        state_2_morning = state_2 + action

                        # If these new states fall outside the range of possible states, then continue
                        if state_1_morning < 0 or state_1_morning > self.max_cars or \
                                state_2_morning < 0 or state_2_morning > self.max_cars:
                            continue

                        # expected return = sum_{s', r} p(s', r|s, a) [r + gamma V(s')]
                        # Use helper function _get_expected_return for this calculation
                        expected_return = self._get_expected_return(state_1_morning, state_2_morning, action)

                        # Want max value (max_a(expected_return) for this state)
                        # Also want to store the action that led to this max value
                        if expected_return > best_return:
                            best_return = expected_return
                            best_action = action

                    # V(s) <- max_a expected return
                    self.value[state_1, state_2] = best_return

                    # Update policy
                    self.policy[state_1, state_2] = best_action

                    # delta <- max(delta, |v - V(s)|)
                    delta = max(delta, np.abs(old_value - self.value[state_1, state_2]))

            print(f"Value improvement: loop {loop_idx}, delta = {delta}")

            # Save the policy and value at the end of each loop
            self.save_artefacts(f"value_iteration_{loop_idx}")

            loop_idx += 1

            if delta < self.theta:
                break

        return self.policy, self.value


if __name__ == "__main__":
    env = JacksCarRental()
    value_iteration = ValueIteration(env)
    policy, value = value_iteration.value_iteration()

    # Plot the policy and value
    plot_policy_and_value(policy, value)
