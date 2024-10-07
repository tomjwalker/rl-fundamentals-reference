"""Scrip which:
- Initialises the gymnasium Blackjack environment
- Randomly samples transitions from the environments
- Plots distributions of the observations"""


import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


# Set up the environment
env = gym.make("Blackjack-v1", sab=True)

# Sample transitions
num_transitions = 10000

states = []
actions = []
rewards = []
next_states = []
dones = []
for _ in range(num_transitions):
    state, info = env.reset()
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)

    states.append(state)
    actions.append(action)
    rewards.append(reward)
    next_states.append(next_state)
    done = terminated or truncated
    dones.append(done)


# Plot the distributions
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# States - separate out into player sum, dealer card, and usable ace
player_sum = [state[0] for state in next_states]
dealer_card = [state[1] for state in next_states]
usable_ace = [state[2] for state in next_states]

axs[0, 0].hist(player_sum, bins=range(4, 32), density=True)
axs[0, 0].set_title("Player Sum")
axs[0, 0].set_xlabel("Player Sum")

axs[0, 1].hist(dealer_card, bins=range(1, 11), density=True)
axs[0, 1].set_title("Dealer Card")
axs[0, 1].set_xlabel("Dealer Card")

axs[1, 0].hist(usable_ace, bins=[0, 1, 2], density=True)
axs[1, 0].set_title("Usable Ace")
axs[1, 0].set_xlabel("Usable Ace")

# Actions
axs[1, 1].hist(actions, bins=[0, 1], density=True)
axs[1, 1].set_title("Actions")
axs[1, 1].set_xlabel("Action")

plt.tight_layout()
plt.show()
