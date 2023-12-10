import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
matplotlib.use('TkAgg')


CWD = os.getcwd()
POLICY_DIR = os.path.join(CWD, ".data", "policy_iteration", "policy")
VALUE_DIR = os.path.join(CWD, ".data", "policy_iteration", "value")


def main():
    for i in range(4):
        policy_filepath = os.path.join(POLICY_DIR, f"policy_improvement_{i}.npy")
        value_filepath = os.path.join(VALUE_DIR, f"policy_evaluation_{i}.npy")
        policy = np.load(policy_filepath)
        value = np.load(value_filepath)
        # print(f"Policy {i}")
        # print(policy)
        # print(f"Value {i}")
        # print(value)
        # print("")

        # Plot the policy: a heatmap of the policy. Use "seismic" colormap centered at 0 (white) to show +ve (red) and
        # -ve (blue) actions
        plt.figure(figsize=(10, 10))
        sns.heatmap(policy, cmap="seismic", center=0, annot=True, fmt=".1f", cbar=False)
        plt.title("Policy")
        plt.xlabel("Location 2")
        plt.ylabel("Location 1")
        plt.title(f"Policy {i}")
        plt.show()


if __name__ == "__main__":
    main()
