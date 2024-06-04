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

    def policy_evaluation(self):
        while True:
            delta = 0
            loop_idx = 0    # For logging training progress
            expected_value_matrix = self.env.get_expected_value(self.value, self.gamma)
            for state_1 in range(self.max_cars + 1):
                for state_2 in range(self.max_cars + 1):

                    # v <- V(s)
                    old_value = self.value[state_1, state_2]

                    # ==================================================================================================
                    # V(s) <- sum_a pi(a|s) sum_{s', r} p(s', r|s, a) [r + gamma V(s')]
                    # ==================================================================================================
                    # a = pi(s) is the action to take (deterministic policy)
                    action = self.policy[state_1, state_2]

                    # Calculate s'. Action a moves cars from location 1 to location 2, so:
                    #    - s'_1 = s_1 - a
                    #    - s'_2 = s_2 + a
                    state_1_next_day = state_1 - action
                    state_2_next_day = state_2 + action

                    # If these new states fall outside the range of possible states, then continue to the next iteration
                    # (don't update the value function for this (state, action) pair)
                    if state_1_next_day < 0 or state_1_next_day > self.max_cars or \
                            state_2_next_day < 0 or state_2_next_day > self.max_cars:
                        continue

                    # TODO: roll this into env class? Somehow make cleaner for homework script
                    # expected return = sum_{s', r} p(s', r|s, a) [r + gamma V(s')]
                    expected_return = expected_value_matrix[state_1_next_day, state_2_next_day] - self.env.move_cost * \
                        np.abs(action)

                    # V(s) <- expected return
                    self.value[state_1, state_2] = expected_return

                    # delta <- max(delta, |v - V(s)|)
                    delta = max(delta, np.abs(old_value - self.value[state_1, state_2]))

            print(f"Policy evaluation: loop {loop_idx}, delta = {delta}")
            loop_idx += 1

            # If delta < self.theta, then the value function has converged, and policy evaluation can stop (break loop)
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        available_actions = np.arange(-self.env.max_move_cars, self.env.max_move_cars + 1)
        expected_value_matrix = self.env.get_expected_value(self.value, self.gamma)
        for state_1 in range(self.max_cars + 1):
            for state_2 in range(self.max_cars + 1):
                old_action = self.policy[state_1, state_2]
                action_returns = [-np.inf] * len(available_actions)
                for action_idx, action in enumerate(available_actions):
                    state_1_next_day = state_1 - action
                    state_2_next_day = state_2 + action
                    # If these new states fall outside the range of possible states, then continue
                    if state_1_next_day < 0 or state_1_next_day > self.max_cars or \
                            state_2_next_day < 0 or state_2_next_day > self.max_cars:
                        continue

                    expected_reward = expected_value_matrix[state_1_next_day, state_2_next_day] - self.env.move_cost * \
                        np.abs(action)
                    action_returns[action_idx] = expected_reward
                self.policy[state_1, state_2] = available_actions[np.argmax(action_returns)]
                if old_action != self.policy[state_1, state_2]:
                    policy_stable = False
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
