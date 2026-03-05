import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_MODULES = [
    "checks.bandits",
    "checks.markov_decision_processes",
    "checks.dynamic_programming",
    "checks.monte_carlo",
    "checks.temporal_difference",
    "checks.planning",
]


@pytest.mark.parametrize("module_name", CHECK_MODULES)
def test_student_checks_cli(module_name: str) -> None:
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")

    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "[PASS]" in result.stdout
    assert "checks passed" in result.stdout.lower()
