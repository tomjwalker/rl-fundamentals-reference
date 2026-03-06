import importlib
import os


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
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)
    original_use = matplotlib.use
    matplotlib.use = lambda *args, **kwargs: None
    try:
        return importlib.import_module(module_name)
    finally:
        matplotlib.use = original_use
