import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class MarkovRewardProcess:
    def __init__(self):
        self.states = ['A', 'B', 'C', 'D', 'E']
        self.terminal_states = ['LH', 'RH']
        self.transition_probs = {
            'A': {'LH': 0.5, 'B': 0.5},
            'B': {'A': 0.5, 'C': 0.5},
            'C': {'B': 0.5, 'D': 0.5},
            'D': {'C': 0.5, 'E': 0.5},
            'E': {'D': 0.5, 'RH': 0.5}
        }
        self.rewards = {
            'LH': 0,
            'RH': 1
        }

    def reset(self):
        return 'C'  # Start state

    def step(self, state):
        next_state = np.random.choice(list(self.transition_probs[state].keys()),
                                      p=list(self.transition_probs[state].values()))
        reward = self.rewards.get(next_state, 0)
        done = next_state in self.terminal_states
        return next_state, reward, done


class TD0Agent:
    def __init__(self, states, alpha=0.1):
        self.alpha = alpha
        self.V = {state: 0.5 for state in states}

    def update(self, state, next_state, reward):
        self.V[state] += self.alpha * (reward + self.V.get(next_state, 0) - self.V[state])


class MCAgent:
    def __init__(self, states, alpha=0.04):  # Changed alpha to 0.04
        self.alpha = alpha
        self.V = {state: 0.5 for state in states}

    def update(self, episode):
        G = 0
        for state, reward in reversed(episode):
            G = reward + G
            self.V[state] += self.alpha * (G - self.V[state])


def run_experiment(num_episodes=100, num_runs=100):
    env = MarkovRewardProcess()
    td_agent = TD0Agent(env.states)
    mc_agent = MCAgent(env.states)

    td_values = np.zeros((num_runs, num_episodes, len(env.states)))
    mc_values = np.zeros((num_runs, num_episodes, len(env.states)))

    true_values = {'A': 1 / 6, 'B': 2 / 6, 'C': 3 / 6, 'D': 4 / 6, 'E': 5 / 6}

    for run in range(num_runs):
        td_agent_run = TD0Agent(env.states)
        mc_agent_run = MCAgent(env.states)

        for episode in range(num_episodes):
            # TD(0) episode
            state = env.reset()
            while True:
                next_state, reward, done = env.step(state)
                td_agent_run.update(state, next_state, reward)
                if done:
                    break
                state = next_state

            # MC episode
            state = env.reset()
            mc_episode = []
            while True:
                next_state, reward, done = env.step(state)
                mc_episode.append((state, reward))
                if done:
                    mc_agent_run.update(mc_episode)
                    break
                state = next_state

            # Store values
            for i, s in enumerate(env.states):
                td_values[run, episode, i] = td_agent_run.V[s]
                mc_values[run, episode, i] = mc_agent_run.V[s]

    return td_values, mc_values, true_values


def plot_results(td_values, mc_values, true_values):
    plt.figure(figsize=(6, 15))

    # Plot 1: Estimated values
    plt.subplot(211)
    states = ['A', 'B', 'C', 'D', 'E']
    true_vals = [true_values[s] for s in states]

    plt.plot(states, true_vals, 'k-', label='True values')
    plt.plot(states, np.mean(td_values[:, -1, :], axis=0), 'b-', label='TD(0)')
    plt.plot(states, np.mean(mc_values[:, -1, :], axis=0), 'r-', label='MC')

    plt.xlabel('State')
    plt.ylabel('Estimated value')
    plt.legend()
    plt.title('Estimated Values')

    # Plot 2: Learning Curves
    plt.subplot(212)
    td_errors = np.sqrt(np.mean((td_values - np.array(true_vals)) ** 2, axis=2))
    mc_errors = np.sqrt(np.mean((mc_values - np.array(true_vals)) ** 2, axis=2))

    plt.plot(np.mean(td_errors, axis=0), 'b-', label='TD(0); α = 0.1')
    plt.plot(np.mean(mc_errors, axis=0), 'r-', label='MC; α = 0.04')

    plt.xlabel('Walks / Episodes')
    plt.ylabel('Empirical RMS error,\naveraged over states')
    plt.legend()
    plt.title('Learning Curves')

    # # Add alpha value annotations
    # plt.annotate(f'TD(0) α = 0.1', xy=(0.7, 0.95), xycoords='axes fraction',
    #              fontsize=9, ha='center', va='center',
    #              bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8))
    # plt.annotate(f'MC α = 0.04', xy=(0.7, 0.87), xycoords='axes fraction',
    #              fontsize=9, ha='center', va='center',
    #              bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8))

    plt.tight_layout(pad=5.0, h_pad=5.0)
    plt.show()


# Run the experiment and plot results
td_values, mc_values, true_values = run_experiment(num_episodes=100, num_runs=100)
plot_results(td_values, mc_values, true_values)
