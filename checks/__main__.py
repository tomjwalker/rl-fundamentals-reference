TOPICS = {
    "bandits": "Check epsilon-greedy selection and update rules.",
    "markov_decision_processes": "Check the analytic Frozen Lake linear-system solutions.",
    "dynamic_programming": "Check policy iteration and value iteration on a small Jack's Car Rental instance.",
    "monte_carlo": "Check Monte Carlo control helpers and one-episode updates.",
    "temporal_difference": "Check Sarsa, Q-learning, and Expected Sarsa on a tiny deterministic task.",
    "planning": "Check Dyna-Q and Dyna-Q+ on a tiny planning task.",
}


def main() -> None:
    print("RL Fundamentals self-check commands")
    print("Use `uv run python -m checks.<topic>` from the repository root.")
    print("If you installed with the pip fallback path, run the same commands without `uv run`.")
    print("")
    for module_name, description in TOPICS.items():
        print(f"- uv run python -m checks.{module_name}")
        print(f"  {description}")


if __name__ == "__main__":
    main()
