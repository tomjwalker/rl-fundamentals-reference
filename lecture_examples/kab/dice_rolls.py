import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# Set random seed for reproducibility
np.random.seed(42)

# Number of die rolls
n_rolls = 1000

# Simulate die rolls (values 1 to 6)
rolls = np.random.randint(1, 7, size=n_rolls)

# Calculate running average
cumulative_sum = np.cumsum(rolls)
roll_numbers = np.arange(1, n_rolls + 1)
running_average = cumulative_sum / roll_numbers

# Theoretical mean of a fair six-sided die
theoretical_mean = 3.5

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(roll_numbers, running_average, label='Observed average', color='#1f77b4')
plt.axhline(y=theoretical_mean, color='#ff7f0e', linestyle='--', label='Theoretical mean')

plt.xlabel('Number of rolls')
plt.ylabel('Average dice roll')
plt.title('Average dice roll by number of rolls')
plt.legend()

# Add grid
plt.grid(True, alpha=0.3)

# Set y-axis limits
plt.ylim(1, 6)

plt.show()