from checks._helpers import OneStepDiscreteEnv, load_module, require, require_close, run_topic


dyna_module = load_module("rl.algorithms.planning.dyna")
dyna_plus_module = load_module("rl.algorithms.planning.dyna_plus")


def check_dyna_model() -> None:
    model = dyna_module.DynaModel(random_seed=0)
    model.add(0, 1, 1.0, 1)
    require(model.get(0, 1) == (1.0, 1), "DynaModel.add() should store the observed transition.")
    require(model.sample_state_action() == (0, 1), "DynaModel.sample_state_action() should be able to sample the stored transition.")


def check_dyna_variants() -> None:
    dyna_agent = dyna_module.Dyna(OneStepDiscreteEnv(), alpha=1.0, gamma=1.0, epsilon=0.0, n_planning_steps=2, random_seed=0)
    dyna_agent.q_values.update(0, 1, 0.5)
    dyna_agent.learn(num_episodes=1)
    require_close(dyna_agent.q_values.get(0, 1), 1.0, message="Dyna should update the chosen action value toward the observed reward.")
    require(dyna_agent.logger.total_rewards_per_episode == [1.0], "Dyna should log one episode reward for one episode of training.")

    dyna_plus_agent = dyna_plus_module.DynaPlus(OneStepDiscreteEnv(), alpha=1.0, gamma=1.0, epsilon=0.0, n_planning_steps=2, random_seed=0)
    dyna_plus_agent.q_values.update(0, 1, 0.5)
    dyna_plus_agent.learn(num_episodes=1)
    require_close(dyna_plus_agent.q_values.get(0, 1), 1.0, message="Dyna-Q+ should update the chosen action value toward the observed reward.")
    require(set(dyna_plus_agent.model.model[0].keys()) == {0, 1}, "Dyna-Q+ should initialise model entries for all actions in a newly seen state.")
    require(dyna_plus_agent.time_since_last_encountered.get(0, 1) == 0, "The visited (state, action) pair should be reset to zero time-since-last-seen.")


def main() -> None:
    run_topic(
        "Planning",
        "Checks the Dyna model helpers and one-episode Dyna-Q / Dyna-Q+ updates.",
        [
            ("Dyna model helper", check_dyna_model),
            ("Dyna-Q and Dyna-Q+", check_dyna_variants),
        ],
    )


if __name__ == "__main__":
    main()
