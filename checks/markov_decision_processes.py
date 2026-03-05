import numpy as np

from checks._helpers import load_module, require, require_close, run_topic


module = load_module("exercises.mdps.frozen_lake_investigation_1")


def check_two_by_two_solution() -> None:
    actual = module.solve_two_by_two()
    require(actual is not None, "solve_two_by_two() should return a NumPy array of state values.")

    gamma = module.GAMMA
    expected = np.linalg.solve(
        np.array([
            [(1 - gamma / 2), (-gamma / 4), (-gamma / 4)],
            [(-gamma / 4), (1 - gamma / 2), 0],
            [(-gamma / 4), 0, (1 - gamma / 2)],
        ]),
        np.array([0, 1 / 4, 1 / 4]),
    )
    require_close(actual, expected, atol=1e-10, message="The 2x2 Frozen Lake linear system is not solved correctly.")


def check_three_by_three_solution() -> None:
    actual = module.solve_three_by_three()
    require(actual is not None, "solve_three_by_three() should return a NumPy array of state values.")

    gamma = module.GAMMA
    expected = np.linalg.solve(
        np.array([
            [(1 - gamma / 2), (-gamma / 4), 0, 0, 0],
            [(-gamma / 4), (1 - gamma / 4), (-gamma / 4), (-gamma / 4), 0],
            [0, (-gamma / 4), (1 - gamma / 2), 0, 0],
            [0, (-gamma / 4), 0, 1, (-gamma / 4)],
            [0, 0, 0, (-gamma / 4), (1 - gamma / 4)],
        ]),
        np.array([0, 0, 0, 0, 1 / 4]),
    )
    require_close(actual, expected, atol=1e-10, message="The 3x3 Frozen Lake linear system is not solved correctly.")


def main() -> None:
    run_topic(
        "Markov decision processes",
        "Checks the analytic Frozen Lake solutions against the expected Bellman-equation systems.",
        [
            ("2x2 Frozen Lake", check_two_by_two_solution),
            ("3x3 Frozen Lake", check_three_by_three_solution),
        ],
    )


if __name__ == "__main__":
    main()
