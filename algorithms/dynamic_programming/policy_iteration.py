import numpy as np
from environment.jacks_car_rental import JacksCarRental
from utils.general import set_filepath


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
        policy_dir = ".data/policy_iteration/policy"
        value_dir = ".data/policy_iteration/value"
        policy_filepath = policy_dir + "/" + save_name + ".npy"
        value_filepath = value_dir + "/" + save_name + ".npy"
        np.save(set_filepath(policy_filepath), self.policy)
        np.save(set_filepath(value_filepath), self.value)

    def policy_evaluation(self):
        while True:
            delta = 0
            loop_idx = 0
            for state_1 in range(self.max_cars + 1):
                for state_2 in range(self.max_cars + 1):
                    old_value = self.value[state_1, state_2]
                    action = self.policy[state_1, state_2]
                    # Initialise expected return with the expected reward. The value component calculated and
                    # appended in the following loop
                    expected_return = self.env.get_expected_reward(
                        state=(state_1, state_2), action=action
                    )
                    for new_state_1 in range(self.max_cars + 1):
                        for new_state_2 in range(self.max_cars + 1):
                            prob_new_state = self.env.get_state_transition_probs(
                                next_state=(new_state_1, new_state_2),
                                state=(state_1, state_2),
                                action=action,
                            )
                            expected_return += (
                                    self.gamma * prob_new_state * self.value[new_state_1, new_state_2]
                            )
                    self.value[state_1, state_2] = expected_return

                    delta = max(delta, np.abs(old_value - self.value[state_1, state_2]))

            print(f"Policy evaluation: loop {loop_idx}, delta = {delta}")
            loop_idx += 1

            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        available_actions = np.arange(-self.env.max_move_cars, self.env.max_move_cars + 1)
        for state_1 in range(self.max_cars + 1):
            for state_2 in range(self.max_cars + 1):
                old_action = self.policy[state_1, state_2]
                action_returns = np.zeros_like(available_actions)
                for action_idx, action in enumerate(available_actions):
                    expected_reward = self.env.get_expected_reward(
                        state=(state_1, state_2), action=action
                    )
                    for new_state_1 in range(self.max_cars + 1):
                        for new_state_2 in range(self.max_cars + 1):
                            prob_new_state = self.env.get_state_transition_probs(
                                next_state=(new_state_1, new_state_2),
                                state=(state_1, state_2),
                                action=action,
                            )
                            expected_reward += (
                                    self.gamma * prob_new_state * self.value[new_state_1, new_state_2]
                            )
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
    print(policy)
    print(value)

    # Plot the policy: a heatmap of the policy. Use "seismic" colormap centered at 0 (white) to show +ve (red) and
    # -ve (blue) actions
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 10))
    sns.heatmap(policy, cmap="seismic", center=0, annot=True, fmt=".1f", cbar=False)
    plt.title("Policy")
    plt.xlabel("Location 2")
    plt.ylabel("Location 1")
    plt.show()

