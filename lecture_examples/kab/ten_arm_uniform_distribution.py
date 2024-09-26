import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


# Number of actions
n_actions = 10

# Probability for each action (uniform distribution)
probabilities = np.ones(n_actions) / n_actions

# Create the plot
fig, ax = plt.subplots(figsize=(5, 2))

# Plot the stems
markerline, stemlines, baseline = ax.stem(range(n_actions), probabilities, basefmt=" ")
plt.setp(markerline, markersize=8)

# Customize the plot
ax.set_xlabel('Action (a)')
ax.set_ylabel('Pr(A_t = a)')
# ax.set_title('Discrete Uniform Distribution for Exploratory Action Selection')
ax.set_xticks(range(n_actions))
ax.set_xticklabels([f'a{i}' for i in range(n_actions)])

# Set y-axis limits
ax.set_ylim(0, max(probabilities) * 1.1)

# Add value labels on top of each stem
for i, p in enumerate(probabilities):
    ax.annotate(f'{p:.2f}', (i, p), xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')

# Add gridlines
ax.grid(True, linestyle='--', alpha=0.7)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()