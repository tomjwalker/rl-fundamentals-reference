import numpy as np

from checks._helpers import load_module, require, require_close, run_topic
from rl.utils.general import argmax_ties_random


class _DummyBandit:
    k = 2


class _DummyEnv:
    bandits = [_DummyBandit()]


def check_argmax_ties_random() -> None:
    np.random.seed(0)
    results = [argmax_ties_random(np.array([0.0, 1.0, 1.0])) for _ in range(40)]
    require(set(results) == {1, 2}, "argmax_ties_random should sample uniformly from all tied best actions.")


def check_epsilon_greedy_updates() -> None:
    module = load_module("rl.algorithms.bandits.epsilon_greedy")

    greedy_agent = module.EpsilonGreedy(_DummyEnv(), epsilon=0.0)
    greedy_agent.q_values = np.array([0.1, 0.9])
    require(greedy_agent.act() == 1, "Greedy action selection should choose the largest estimated value.")

    sample_avg_agent = module.EpsilonGreedy(_DummyEnv(), epsilon=0.0)
    sample_avg_agent.simple_update(0, 1.0)
    require_close(sample_avg_agent.q_values[0], 1.0, message="Sample-average update should match the incremental formula.")

    weighted_agent = module.EpsilonGreedy(_DummyEnv(), epsilon=0.0, alpha=0.25)
    weighted_agent.weighted_update(0, 1.0)
    require_close(weighted_agent.q_values[0], 0.25, message="Weighted update should use the configured alpha step size.")


def main() -> None:
    run_topic(
        "Bandits",
        "Checks the tie-breaking helper and the core epsilon-greedy update rules.",
        [
            ("Random tie-breaking", check_argmax_ties_random),
            ("Epsilon-greedy updates", check_epsilon_greedy_updates),
        ],
    )


if __name__ == "__main__":
    main()
