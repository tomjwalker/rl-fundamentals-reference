from checks._helpers import OneStepTupleEnv, load_module, require, require_close, run_topic
from rl.common.policy import EpsilonGreedyPolicy
from rl.common.q_value_table import QValueTable


on_policy_module = load_module("rl.algorithms.monte_carlo.on_policy")
off_policy_module = load_module("rl.algorithms.monte_carlo.off_policy")


def check_tied_action_probabilities() -> None:
    q_values = QValueTable((1,), 2)
    q_values.update(0, 0, 1.0)
    q_values.update(0, 1, 1.0)

    policy = EpsilonGreedyPolicy(epsilon=0.2, action_space=2)
    probs = policy.compute_probs(0, q_values)
    require_close(probs, [0.5, 0.5], message="Tied greedy actions should split the greedy mass evenly.")


def check_on_policy_update() -> None:
    agent = on_policy_module.MCOnPolicy(OneStepTupleEnv(), gamma=1.0, epsilon=0.0, random_seed=0)
    agent.q_values.update((0, 0, 0), 1, 0.5)
    agent.learn(num_episodes=1)

    require_close(agent.q_values.get((0, 0, 0), 1), 1.0, message="MC On-Policy should move the visited action value toward the observed return.")
    require(agent.logger.total_rewards_per_episode == [1.0], "MC On-Policy should log one episode reward for one episode of training.")


def check_off_policy_update() -> None:
    agent = off_policy_module.MCOffPolicy(OneStepTupleEnv(), gamma=1.0, epsilon=0.0, random_seed=0)
    agent.q_values.update((0, 0, 0), 1, 0.5)
    agent.learn(num_episodes=1)

    require_close(agent.q_values.get((0, 0, 0), 1), 1.0, message="MC Off-Policy should update the visited action value from the observed return.")
    require(agent.logger.total_rewards_per_episode == [1.0], "MC Off-Policy should log one episode reward for one episode of training.")


def main() -> None:
    run_topic(
        "Monte Carlo",
        "Checks the epsilon-greedy probability helper and one-episode Monte Carlo control updates.",
        [
            ("Tied-action probabilities", check_tied_action_probabilities),
            ("On-policy Monte Carlo", check_on_policy_update),
            ("Off-policy Monte Carlo", check_off_policy_update),
        ],
    )


if __name__ == "__main__":
    main()
