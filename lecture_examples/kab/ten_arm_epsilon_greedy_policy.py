import numpy as np
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use('TkAgg')

# Parameters
n_actions = 10
epsilon = 0.25

# Create probabilities for ε-greedy policy
probabilities = np.full(n_actions, epsilon / n_actions)
greedy_action = 4
probabilities[greedy_action] += 1 - epsilon

# Create the plot
fig, ax = plt.subplots(figsize=(6, 3))

# Plot the stems
markerline, stemlines, baseline = ax.stem(range(n_actions), probabilities, basefmt=" ")
plt.setp(markerline, markersize=8)

# Customize the plot
ax.set_xlabel('Action (a)', fontsize=12)
ax.set_ylabel('π(a) = Pr(A_t = a)', fontsize=12)
ax.set_title(f'ε-greedy Policy Distribution (ε = {epsilon})', fontsize=12)
ax.set_xticks(range(n_actions))
ax.set_xticklabels([f'a{i}' for i in range(n_actions)])

# Set y-axis limits
ax.set_ylim(0, max(probabilities) * 1.1)

# Add value labels on top of each stem
for i, p in enumerate(probabilities):
    ax.annotate(f'{p:.3f}', (i, p), xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')

# Add gridlines
ax.grid(True, linestyle='--', alpha=0.7)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#
# # Highlight the greedy action
# ax.get_children()[greedy_action * 2 + 1].set_color('r')  # Stem
# ax.get_children()[greedy_action * 2 + 2].set_color('r')  # Marker
#
# # Add a legend
# ax.legend(['Explore', 'Exploit (Greedy Action)'], loc='upper right')

plt.tight_layout()
plt.show()