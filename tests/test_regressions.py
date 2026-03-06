import os
from types import SimpleNamespace

import numpy as np

from rl.common.policy import EpsilonGreedyPolicy
from rl.common.q_value_table import QValueTable
from tests.support import OneStepTupleEnv, TwoStepTupleEnv, load_module


mc_agent_module = load_module("rl.algorithms.common.mc_agent")
off_policy_module = load_module("rl.algorithms.monte_carlo.off_policy")
viz_module = load_module("rl.algorithms.monte_carlo.viz")
value_iteration_module = load_module("rl.algorithms.dynamic_programming.value_iteration")


class DummyMonteCarloAgent(mc_agent_module.MonteCarloAgent):
    def act(self, state):
        return 1

    def learn(self, num_episodes: int = 1) -> None:
        raise NotImplementedError


def test_generate_episode_logs_each_timestep() -> None:
    agent = DummyMonteCarloAgent(TwoStepTupleEnv(), gamma=1.0)

    episode = agent._generate_episode(exploring_starts=False)
    agent.logger.log_episode()

    assert len(episode) == 2
    assert agent.logger.cumulative_rewards == [0.0, 1.0]
    assert agent.logger.steps_per_episode == [2]
    assert agent.logger.total_rewards_per_episode == [1.0]


def test_off_policy_logs_episode_level_reward() -> None:
    agent = off_policy_module.MCOffPolicy(OneStepTupleEnv(), gamma=1.0, epsilon=0.0, random_seed=0)
    agent.q_values.update((0, 0, 0), 1, 0.5)

    agent.learn(num_episodes=1)

    assert agent.logger.total_rewards_per_episode == [1.0]
    assert agent.logger.steps_per_episode == [1]


def test_monte_carlo_viz_reads_q_value_table_values() -> None:
    q_values = np.zeros((22, 11, 2, 2))
    q_values[:, :, 1, 1] = 1.0
    agent = SimpleNamespace(name="MC On-Policy", q_values=SimpleNamespace(values=q_values))

    policy = viz_module._get_policy_for_agent(agent, usable_ace=True)

    assert policy.shape == (11, 10)
    assert np.all(policy == 1)


def test_epsilon_greedy_probabilities_split_ties_evenly() -> None:
    q_values = QValueTable((1,), 2)
    q_values.update(0, 0, 1.0)
    q_values.update(0, 1, 1.0)

    policy = EpsilonGreedyPolicy(epsilon=0.2, action_space=2)
    probs = policy.compute_probs(0, q_values)

    assert np.allclose(probs, np.array([0.5, 0.5]))


def test_value_iteration_saves_to_value_iteration_tree(tmp_path) -> None:
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        env = value_iteration_module.JacksCarRental(max_cars=1, max_move_cars=1)
        solver = value_iteration_module.ValueIteration(env)
        solver.save_artefacts("smoke")
    finally:
        os.chdir(old_cwd)

    assert (tmp_path / ".data" / "dynamic_programming" / "value_iteration" / "policy" / "smoke.npy").is_file()
    assert (tmp_path / ".data" / "dynamic_programming" / "value_iteration" / "value" / "smoke.npy").is_file()
    assert not (tmp_path / ".data" / "dynamic_programming" / "policy_iteration").exists()
