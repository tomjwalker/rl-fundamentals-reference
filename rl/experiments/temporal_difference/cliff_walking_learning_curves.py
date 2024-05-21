from rl.algorithms.temporal_difference.q_learning import QLearning
from rl.algorithms.temporal_difference.expected_sarsa import ExpectedSarsa
from rl.algorithms.temporal_difference.sarsa import Sarsa
from rl.simulation.trial import Trial
from rl.environment.temporal_difference.cliff_walking_utils import visualise_q    # Specific visualisation function for cliff walking    # NoQA
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt


# Trial parameters
sessions = 10
episodes_per_session = 500

# Set up the environment
env = gym.make("CliffWalking-v0")

# Set up the agents
agents = {"Sarsa": (Sarsa, "blue"), "Q-Learning": (QLearning, "red"), "Expected Sarsa": (ExpectedSarsa, "green")}

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 5))

# Run the trials
for agent_name, (agent_class, plot_color) in agents.items():

    # Initialise the trial
    trial = Trial(
        agent_class,
        env,
        sessions=sessions,
        episodes_per_session=episodes_per_session,
        random_seeds=np.arange(sessions)
    )

    # Run the trial
    trial.run()

    # Plot the results
    trial.plot(color=plot_color, ax=ax, show_std=False)

plt.title("Cliff Walking: Learning Curves")
plt.ylim([-100, 0])
plt.legend()
plt.show()
