import importlib
import os
from typing import Callable, Iterable

import numpy as np


class CheckFailure(Exception):
    """Raised when a student-facing self-check fails."""


class SimpleSpace:
    def __init__(self, n: int) -> None:
        self.n = n


class OneStepDiscreteEnv:
    observation_space = SimpleSpace(2)
    action_space = SimpleSpace(2)

    def reset(self):
        return 0, {}

    def step(self, action: int):
        reward = 1.0 if action == 1 else 0.0
        return 1, reward, True, False, {}


class OneStepTupleEnv:
    observation_space = [SimpleSpace(2), SimpleSpace(2), SimpleSpace(2)]
    action_space = SimpleSpace(2)

    def reset(self):
        return (0, 0, 0), {}

    def step(self, action: int):
        reward = 1.0 if action == 1 else 0.0
        return (1, 0, 0), reward, True, False, {}


class TwoStepTupleEnv:
    observation_space = [SimpleSpace(2), SimpleSpace(2), SimpleSpace(2)]
    action_space = SimpleSpace(2)

    def __init__(self) -> None:
        self._step = 0

    def reset(self):
        self._step = 0
        return (0, 0, 0), {}

    def step(self, action: int):
        if self._step == 0:
            self._step += 1
            return (1, 0, 0), 0.0, False, False, {}

        reward = 1.0 if action == 1 else 0.0
        return (1, 1, 0), reward, True, False, {}


def load_module(module_name: str):
    """Import modules that hard-code a plotting backend without letting them force Tk."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)
    original_use = matplotlib.use
    matplotlib.use = lambda *args, **kwargs: None
    try:
        return importlib.import_module(module_name)
    finally:
        matplotlib.use = original_use


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def require_close(actual, expected, atol: float = 1e-8, rtol: float = 1e-7, message: str = "") -> None:
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise CheckFailure(message or f"Expected {expected!r}, got {actual!r}.")


def run_topic(title: str, description: str, checks: Iterable[tuple[str, Callable[[], None]]]) -> None:
    print(f"{title} self-check")
    print(description)
    print("")

    failures: list[tuple[str, str]] = []
    total = 0
    for label, check in checks:
        total += 1
        try:
            check()
            print(f"[PASS] {label}")
        except Exception as exc:  # noqa: BLE001
            failures.append((label, str(exc)))
            print(f"[FAIL] {label}: {exc}")

    print("")
    if failures:
        print(f"{len(failures)} of {total} checks failed. Fix the items above and rerun this command.")
        raise SystemExit(1)

    print(f"All {total} checks passed.")
