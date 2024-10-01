# TODO: read then remove this: https://alexkozlov.com/post/jack-car-rental/
# TODO: compare Policy and Value iteration: computation time, number of iterations, etc.
# TODO: central experiment script


import numpy as np
from rl.environment.dynamic_programming.jacks_car_rental import JacksCarRental
from rl.utils.general import set_filepath
import os
from rl.algorithms.dynamic_programming.viz import plot_policy_and_value


class PolicyIteration:
    def __init__(self, env, gamma=0.9, theta=1e-8):
        self.env = env
        self.gamma = gamma
        self.theta = theta

        # TODO: might need to return to this if env refactored following Gymnasium API
        self.max_cars = env.max_cars
        self.policy = np.zeros((self.max_cars + 1, self.max_cars + 1), dtype=np.int8)    # int8 fits action range
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

    def policy_evaluation(self):

        # Lecture algorithm pseudo-code: "repeat:"
        while True:

            # HOMEWORK: delta <- 0
            delta = 0

            # Efficiently calculate expected returns for all states
            self._update_expected_return_array()

            for state_1 in range(self.max_cars + 1):
                for state_2 in range(self.max_cars + 1):

                    # HOMEWORK: store old value ("v <- V(s)"). c.f. self.value
                    old_value = self.value[state_1, state_2]

                    # HOMEWORK: retrieve a <- pi(s) as the action to take (deterministic policy). c.f. self.policy
                    action = self.policy[state_1, state_2]

                    # HOMEWORK: the environment object has a method that computes the next state given the current state
                    # and the action.
                    # Use this method to compute the next state: next_state = env.compute_next_state(...)
                    next_state = self.env.compute_next_state((state_1, state_2), action)

                    # The next state might be invalid (e.g. if the action is to move more cars than are available.
                    # The environment method returns None in this case, and this control structure skips the rest of the
                    # loop iteration.
                    if next_state is None:
                        continue  # Skip invalid states

                    # Unpack the next state
                    next_state_1, next_state_2 = next_state

                    # HOMEWORK: Use helper function _get_expected_return for this step
                    # This calculates "expected return = sum_{s', r} p(s', r|s, a) [r + gamma V(s')]" efficiently
                    expected_return = self._get_expected_return(next_state_1, next_state_2, action)

                    # HOMEWORK: V(s) <- expected return
                    self.value[state_1, state_2] = expected_return

                    # HOMEWORK: delta <- max(delta, |v - V(s)|)
                    delta = max(delta, np.abs(old_value - self.value[state_1, state_2]))

            # HOMEWORK START: (2 lines)
            # If delta < self.theta, then the value function has converged, and policy evaluation can stop (break loop)
            if delta < self.theta:
                break
            # HOMEWORK END

    def policy_improvement(self):

        # HOMEWORK: policy_stable <- True
        policy_stable = True

        # Initialise available actions
        available_actions = np.arange(-self.env.max_move_cars, self.env.max_move_cars + 1)

        # For each s in S
        for state_1 in range(self.max_cars + 1):
            for state_2 in range(self.max_cars + 1):

                # HOMEWORK: store old action ("old_action <- pi(s)")
                old_action = self.policy[state_1, state_2]

                # ======================================================================================================
                # pi(s) <- argmax_a sum_{s', r} p(s', r|s, a) [r + gamma V(s')]
                # ======================================================================================================

                # Initialise action_returns for all possible actions as [-inf, -inf, ..., -inf]
                # Each element in action_returns corresponds to an action in available_actions (running from
                # [-max_move_cars, max_move_cars] (inclusive))
                # This will make the argmax calculation easier (list elements will be replaced by expected rewards for
                # each action)
                action_returns = []

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
                    expected_return = self._get_expected_return(next_state_1, next_state_2, action)

                    # HOMEWORK: Update action_returns list with expected return for this action
                    action_returns.append(expected_return)

                # HOMEWORK: Once all actions have been evaluated, select the best action
                # (use np.argmax on action_returns to find the index for the best action in available_actions)
                best_action = available_actions[np.argmax(action_returns)]

                # HOMEWORK: Update policy with best action
                self.policy[state_1, state_2] = best_action

                # HOMEWORK START: (2 lines)
                # If old_action != pi(s), then policy_stable <- False
                if old_action != self.policy[state_1, state_2]:
                    policy_stable = False
                # HOMEWORK END

        return policy_stable

    def policy_iteration(self):
        loop = 0
        while True:

            print(f"Policy evaluation: loop {loop}")
            self.policy_evaluation()
            self.save_artefacts(save_name=f"policy_evaluation_{loop}")

            print(f"Policy improvement: loop {loop}")
            policy_stable = self.policy_improvement()
            self.save_artefacts(save_name=f"policy_improvement_{loop}")

            if policy_stable:
                break
            loop += 1
        return self.policy, self.value


if __name__ == "__main__":
    env = JacksCarRental()
    policy_iteration = PolicyIteration(env)
    policy, value = policy_iteration.policy_iteration()

    # Plot the policy and value
    plot_policy_and_value(policy, value)
