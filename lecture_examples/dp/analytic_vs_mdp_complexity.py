import numpy as np

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


# Define the range of n (number of states)
n_values = np.arange(1, 101, 1)

# Calculate time and space complexity for analytic methods
analytic_time = n_values**3
analytic_space = n_values**2

# Calculate time and space complexity for dynamic programming (assuming k=10 iterations)
dp_time = n_values**2 * 10
dp_space = n_values

# Create the subplot
fig, axes = plt.subplots(2, 1, figsize=(7, 12))

# Plot time complexity on the left
axes[0].plot(n_values, analytic_time, label="Analytic (O(n³))", color="blue")
axes[0].plot(n_values, dp_time, label="Dynamic Programming (O(n²k))", color="orange")
axes[0].set_xlabel("Number of States (n)")
axes[0].set_ylabel("Time Complexity")
axes[0].set_title("Time Complexity Comparison")
axes[0].legend()
axes[0].grid(True)

# Plot space complexity on the right
axes[1].plot(n_values, analytic_space, label="Analytic (O(n²))", color="blue")
axes[1].plot(n_values, dp_space, label="Dynamic Programming (O(n))", color="orange")
axes[1].set_xlabel("Number of States (n)")
axes[1].set_ylabel("Space Complexity")
axes[1].set_title("Space Complexity Comparison")
axes[1].legend()
axes[1].grid(True)

plt.subplots_adjust(hspace=0.5)

# plt.tight_layout()
plt.show()
