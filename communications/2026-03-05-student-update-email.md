# Student Update Email

Subject: Course repo refresh: Python 3.11, simpler setup, self-checks, and bug fixes

Hi everyone,

I have just shipped a course repo refresh for *Introduction to Reinforcement Learning*.

This is not a redesign of the course or a move away from hand-coding. The aim was to make the repos easier to run in 2026, add better ways to check your own work locally, and fix a small number of confirmed bugs and rough edges in the reference code.

## What changed

### 1. The repo now has a clear Python target

The course repos now target **Python 3.11**.

That gives a version that is modern enough to be a sensible default in 2026, but still compatible with the conservative package versions used in the course.

### 2. Setup is now `uv`-first, with a pip fallback

The reference repo now includes:

- `pyproject.toml`
- `uv.lock`
- `.python-version`
- `requirements.txt`

Recommended setup is now:

```bash
uv python install 3.11
uv sync --python 3.11
```

If you do not want to use `uv`, the repo still supports the normal `venv` + `pip install -r requirements.txt` path.

### 3. You no longer need to touch `PYTHONPATH`

The README and assignment instructions now use module-based commands from the repo root, so setup is less fiddly and less error-prone.

### 4. New beginner-friendly local self-checks

I added a `checks/` package so you can validate your work more directly than just comparing against the reference repo or eyeballing plots.

You can now run commands like:

```bash
uv run python -m checks.bandits
uv run python -m checks.markov_decision_processes
uv run python -m checks.dynamic_programming
uv run python -m checks.monte_carlo
uv run python -m checks.temporal_difference
uv run python -m checks.planning
```

These checks are designed to be lightweight and beginner-friendly:

- exact checks where exact answers make sense
- seeded/tolerant checks where randomness is involved
- human-readable pass/fail output
- no need for a Coursera-style grader to get useful feedback

### 5. Student repo generation has been improved

The script that generates the beginner and advanced student repos now copies the new setup files and the local checks as well.

It also fixes a marker bug so both `# HOMEWORK START:` and `# HOMEWORK STARTS:` blocks are handled properly.

### 6. Assignment docs have been clarified

I updated the assignment notes in a few places that were causing confusion:

- Temporal Difference now explains the difference between the SARSA, Q-learning, and Expected SARSA targets more directly.
- Frozen Lake now explicitly tells you to work inside `ASSIGNMENT START/END` sections and ignore `SOLUTION` blocks in the reference repo.
- The pentagram Monte Carlo exercise is now framed as optional/bonus content rather than core path material.
- A bad pentagram file reference was corrected.

## Confirmed bugs that were fixed

I only fixed items I was confident were genuinely wrong.

### Monte Carlo

- Monte Carlo timestep logging was fixed so rewards and step counts are logged per environment step rather than only once at the end.
- Off-policy Monte Carlo now logs episode-level results properly.
- Monte Carlo Blackjack visualisation was fixed to read the correct Q-value table attribute.

### Policies and TD/MC internals

- Epsilon-greedy action probabilities now handle tied greedy actions correctly.
  - This matters in Expected SARSA and off-policy Monte Carlo.

### Bandits

- A reproducibility bug in the bandit experiments was fixed so seeding behaves as intended.

### Dynamic Programming

- Value iteration artefacts now save into the correct `value_iteration` directory tree rather than the `policy_iteration` tree.

### Planning utilities

- A broken legacy planning-agent import was fixed.

## What this means for you as a student

If you are starting fresh, you should use the updated repo and follow the new Python 3.11 setup instructions.

If you already have a local copy of one of the student repos:

- you do **not** need to throw away your work
- but it is worth pulling the latest version once I have pushed the updated student repos
- if you hit old setup issues, Python-version confusion, or unclear assignment guidance, the refresh should help

## What has *not* changed

- The course is still hand-coding focused.
- I have not turned the assignments into auto-generated or AI-completed exercises.
- I have not done a broad refactor or mass formatting pass over the repo.
- The intent is still for the code to stay readable as a teaching resource, not to turn it into an over-engineered framework.

## Thanks

A number of these fixes came from student feedback over the last year, especially around setup friction, unclear assignment instructions, and the Monte Carlo visualisation issue.

Thanks to everyone who took the time to report problems or suggest improvements.

Best,
Tom