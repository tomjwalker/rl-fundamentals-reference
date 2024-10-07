# TODO: hyperparameter sweep as pp. 133 of Sutton and Barto
# TODO: state visitation count plots

from rl.algorithms.temporal_difference.q_learning import QLearning
from rl.algorithms.temporal_difference.expected_sarsa import ExpectedSarsa
from rl.algorithms.temporal_difference.sarsa import Sarsa
from rl.simulation.trial import Trial
from rl.environment.temporal_difference.cliff_walking_utils import visualise_q
# from rl.environment.temporal_difference.cliff_walking_utils import visualise_state_visits    # Specific visualisation function for cliff walking    # NoQA
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt


def run(sessions: int, episodes_per_session: int, show_state_visits: bool = False) -> None:
    """
    Runs the trials for different TD agents on the CliffWalking environment and plots the results.

    Args:
        sessions (int): Number of sessions for each agent.
        episodes_per_session (int): Number of episodes per session for each agent.
        show_state_visits (bool, optional): Flag to show state visitation counts.
    """
    # Enable interactive mode
    plt.ion()

    # Set up the environment
    env = gym.make("CliffWalking-v0")

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run TD agents on the CliffWalking environment and plot the results.")
    parser.add_argument(
        '--sessions',
        type=int,
        default=30,
        help="Number of sessions for each agent."
    )
    parser.add_argument(
        '--episodes_per_session',
        type=int,
        default=500,
        help="Number of episodes per session for each agent."
    )
    parser.add_argument(
        '--show_state_visits',
        action='store_true',
        help="Flag to show state visitation counts."
    )
    args = parser.parse_args()

    run(
        sessions=args.sessions,
        episodes_per_session=args.episodes_per_session,
        show_state_visits=args.show_state_visits
    )
