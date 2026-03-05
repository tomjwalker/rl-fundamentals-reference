from checks._helpers import OneStepDiscreteEnv, require, require_close, run_topic
from rl.algorithms.temporal_difference.expected_sarsa import ExpectedSarsa
from rl.algorithms.temporal_difference.q_learning import QLearning
from rl.algorithms.temporal_difference.sarsa import Sarsa


AGENTS = [Sarsa, QLearning, ExpectedSarsa]


def check_temporal_difference_updates() -> None:
    for agent_class in AGENTS:
        agent = agent_class(OneStepDiscreteEnv(), alpha=1.0, gamma=1.0, epsilon=0.0, random_seed=0)
        agent.q_values.update(0, 1, 0.5)
        agent.learn(num_episodes=1)

        require_close(
            agent.q_values.get(0, 1),
            1.0,
            message=f"{agent_class.__name__} should update the chosen action toward the observed reward on the one-step test task.",
        )
        require(
            agent.logger.total_rewards_per_episode == [1.0],
            f"{agent_class.__name__} should log one episode reward for one episode of training.",
        )


def main() -> None:
    run_topic(
        "Temporal difference",
        "Checks Sarsa, Q-learning, and Expected Sarsa on a tiny deterministic task.",
        [("TD one-step updates", check_temporal_difference_updates)],
    )


if __name__ == "__main__":
    main()
