import numpy as np
from environment.jacks_car_rental import JacksCarRental
from utils.general import set_filepath


class ValueIteration:
    def __init__(self, env, gamma=0.9, theta=1e-8):
        self.env = env
        self.gamma = gamma
        self.theta = theta

        # TODO: might need to return to this if env refactored following Gymnasium API
        self.max_cars = env.max_cars
        self.policy = np.zeros((self.max_cars + 1, self.max_cars + 1), dtype=np.int8)    # int8 fits action range
        self.value = np.zeros((self.max_cars + 1, self.max_cars + 1), dtype=np.float32)

    def save_artefacts(self, save_name):
        policy_dir = ".data/dynamic_programming/value_iteration/policy"
        value_dir = ".data/dynamic_programming/value_iteration/value"
        policy_filepath = policy_dir + "/" + save_name + ".npy"
        value_filepath = value_dir + "/" + save_name + ".npy"
        np.save(set_filepath(policy_filepath), self.policy)
        np.save(set_filepath(value_filepath), self.value)

    def value_iteration(self):
        loop_idx = 0
        while True:
            delta = 0
            expected_value_matrix = self.env.get_expected_value(self.value, self.gamma)
            available_actions = np.arange(-self.env.max_move_cars, self.env.max_move_cars + 1)
            for state_1 in range(self.max_cars + 1):
                for state_2 in range(self.max_cars + 1):
                    # Initialise {best_action, best_return} dict for this state
                    best_action = None
                    best_return = -np.inf
                    old_value = self.value[state_1, state_2]
                    for action in available_actions:
                        state_1_next_day = state_1 - action
                        state_2_next_day = state_2 + action
                        # If these new states fall outside the range of possible states, then continue
                        if state_1_next_day < 0 or state_1_next_day > self.max_cars or \
                                state_2_next_day < 0 or state_2_next_day > self.max_cars:
                            continue

                        # TODO: roll this into env class? Somehow make cleaner for homework script
                        expected_return = expected_value_matrix[state_1_next_day, state_2_next_day] - \
                                          self.env.move_cost * np.abs(action)

                        if expected_return > best_return:
                            best_action = action
                            best_return = expected_return

                    self.value[state_1, state_2] = best_return
                    self.policy[state_1, state_2] = best_action
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
