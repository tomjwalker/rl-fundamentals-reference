import os

import numpy as np

from rl.algorithms.temporal_difference.q_learning import QLearning
from tests.support import OneStepDiscreteEnv, OneStepTupleEnv, load_module


bandit_module = load_module("rl.algorithms.bandits.epsilon_greedy")
monte_carlo_module = load_module("rl.algorithms.monte_carlo.on_policy")
planning_module = load_module("rl.algorithms.planning.dyna")
value_iteration_module = load_module("rl.algorithms.dynamic_programming.value_iteration")
planning_agent_module = load_module("rl.algorithms.common.planning_agent")


def test_bandits_smoke() -> None:
    env = bandit_module.KArmedTestbed(num_runs=2, k=2, k_mean=0, k_std=1, bandit_std=1, with_seed=True)
    agent = bandit_module.EpsilonGreedy(env, epsilon=0.1, max_steps=5)

    rewards, optimal = agent.train()

    assert rewards.shape == (5, 2)
    assert optimal.shape == (5, 2)
    assert np.isfinite(rewards.to_numpy()).all()


def test_temporal_difference_smoke() -> None:
    agent = QLearning(OneStepDiscreteEnv(), alpha=1.0, gamma=1.0, epsilon=0.0, random_seed=0)
    agent.learn(num_episodes=2)

    assert agent.logger.total_rewards_per_episode == [1.0, 1.0]


def test_monte_carlo_smoke() -> None:
    agent = monte_carlo_module.MCOnPolicy(OneStepTupleEnv(), gamma=1.0, epsilon=0.0, random_seed=0)
    agent.learn(num_episodes=2)

    assert len(agent.logger.total_rewards_per_episode) == 2
    assert np.isfinite(agent.q_values.values).all()


def test_planning_smoke() -> None:
    agent = planning_module.Dyna(OneStepDiscreteEnv(), alpha=1.0, gamma=1.0, epsilon=0.0, n_planning_steps=2, random_seed=0)
    agent.learn(num_episodes=2)

    assert agent.logger.total_rewards_per_episode == [1.0, 1.0]


def test_dynamic_programming_smoke(tmp_path) -> None:
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        env = value_iteration_module.JacksCarRental(max_cars=1, max_move_cars=1)
        solver = value_iteration_module.ValueIteration(env, theta=1e-4)
        policy, value = solver.value_iteration()
    finally:
        os.chdir(old_cwd)

    assert policy.shape == (2, 2)
    assert value.shape == (2, 2)
    assert np.isfinite(value).all()


def test_planning_agent_module_imports() -> None:
    assert planning_agent_module.PlanningAgent.__name__ == "PlanningAgent"
