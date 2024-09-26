# TODO: hyperparameter sweep as pp. 133 of Sutton and Barto
# TODO: state visitation count plots


from rl.algorithms.temporal_difference.q_learning import QLearning
from rl.algorithms.temporal_difference.expected_sarsa import ExpectedSarsa
from rl.algorithms.temporal_difference.sarsa import Sarsa
from rl.simulation.trial import Trial
from rl.environment.temporal_difference.cliff_walking_utils import visualise_q
# from rl.environment.temporal_difference.cliff_walking_utils import visualise_state_visits    # Specific visualisation
# function for cliff walking    # NoQA
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt


# Enable interactive mode
plt.ion()


# Trial parameters
sessions = 30
episodes_per_session = 500

# Set up the environment
env = gym.make("CliffWalking-v0")

# Show state visits plot flag (doubles the calculation time)
# TODO: fold in state visit method to current approach
show_state_visits = False

# Set up the agents
agents = {"Sarsa": (Sarsa, "blue"), "Q-Learning": (QLearning, "red"), "Expected Sarsa": (ExpectedSarsa, "green")}

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 5))

# Run the trials
trained_agent_last_run = {}
for agent_name, (agent_class, plot_color) in agents.items():

    # Initialise the trial
    trial = Trial(
        agent_class=agent_class,
        environment=env,
        sessions=sessions,
        episodes_per_session=episodes_per_session,
        random_seeds=np.arange(sessions)
    )

    # Run the trial
    trial.run()

    # Plot the results
    trial.plot(
        series_type="total_rewards_per_episode",
        color=plot_color,
        ax=ax,
        show_std=False
    )

    # Store the last run's trained agent
    trained_agent_last_run[agent_name] = trial.agent

# Plot learning curves
plt.title("Cliff Walking: Learning Curves")
plt.ylim([-100, 0])
plt.legend()
plt.show()

# Plot the Q-values
fig, ax = plt.subplots(len(agents), 1, figsize=(10, 5 * len(agents)))
for i, (agent_name, _) in enumerate(agents.items()):
    visualise_q(trained_agent_last_run[agent_name], ax=ax[i])
plt.show()

if show_state_visits:

    # TODO: check this is fully implemented correctly

    # Track state visitations
    state_visits = {agent_name: np.zeros(env.observation_space.n) for agent_name in agents}


# Keep the plot open
plt.ioff()

# Wait for user to close the plot
plt.show()

# Close the plot
plt.close()
