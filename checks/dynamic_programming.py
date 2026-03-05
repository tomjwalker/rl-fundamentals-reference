import os
import tempfile

import numpy as np

from checks._helpers import load_module, require, require_close, run_topic


policy_module = load_module("rl.algorithms.dynamic_programming.policy_iteration")
value_module = load_module("rl.algorithms.dynamic_programming.value_iteration")


def check_small_jacks_car_rental() -> None:
    env = value_module.JacksCarRental(max_cars=2, max_move_cars=1)

    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        try:
            policy_iteration = policy_module.PolicyIteration(env, theta=1e-4)
            value_iteration = value_module.ValueIteration(env, theta=1e-4)

            policy_pi, value_pi = policy_iteration.policy_iteration()
            policy_vi, value_vi = value_iteration.value_iteration()
        finally:
            os.chdir(old_cwd)

    require(policy_pi.shape == (3, 3), "Policy iteration should return a 3x3 policy on the small test environment.")
    require(policy_vi.shape == (3, 3), "Value iteration should return a 3x3 policy on the small test environment.")
    require(np.isfinite(value_pi).all() and np.isfinite(value_vi).all(), "State values should stay finite.")
    require(policy_pi[0, 0] == 0 and policy_vi[0, 0] == 0, "At state (0, 0) the best action should be to move no cars.")
    require(value_vi[2, 2] > value_vi[0, 0], "States with more cars available should be more valuable on the small test environment.")
    require(np.array_equal(policy_pi, policy_vi), "Policy iteration and value iteration should agree on the small test environment.")
    require_close(value_pi, value_vi, atol=1e-2, message="Policy iteration and value iteration should produce similar value functions.")


def main() -> None:
    run_topic(
        "Dynamic programming",
        "Checks policy iteration and value iteration on a small Jack's Car Rental instance.",
        [("Small Jack's Car Rental run", check_small_jacks_car_rental)],
    )


if __name__ == "__main__":
    main()
