"""
TODO:
    - Figure out strange policy plots
"""

from rl.algorithms.monte_carlo.viz import plot_results
from rl.algorithms.common.mc_agent import MonteCarloAgent
# N.B., numpy argmax used in this instance to ensure ties are broken consistently

from rl.common.policy import EpsilonGreedyPolicy, DeterministicPolicy

import gymnasium as gym
import numpy as np

from typing import Union
from gymnasium import Env
from rl.common.results_logger import ResultsLogger

import matplotlib
matplotlib.use('TkAgg')
#
#
# # TODO: move this to utils (as with other MC modules)
# def _is_subelement_present(subelement, my_list):
#     """
#     Helps check if a subelement is present in a list of tuples. Used to check if state has already been seen.
#
#     Simple example:
#     _is_subelement_present((1, 2), [(1, 2, 3), (4, 5, 6)])
#         True
#     """
#     for tpl in my_list:
#         if subelement == tpl[:len(subelement)]:
#             return True
#     return False


class MCOffPolicy(MonteCarloAgent):

    def __init__(
            self,
            env: Union[Env, object],
            gamma: float,
            epsilon: float = None,
            logger: ResultsLogger = None,
            random_seed: int = None,
    ):

        super().__init__(env, gamma, epsilon, logger, random_seed)

        self.name = "MC Off-Policy"    # For plotting

        self.policy = None
        self.behaviour_policy = None
        self.reset()

    def reset(self):

        super().reset()

        # Initialise target policy (deterministic)
        self.policy = DeterministicPolicy(self.state_shape)

        # Initialise behaviour policy (epsilon-greedy)
        self.behaviour_policy = EpsilonGreedyPolicy(self.epsilon, self.env.action_space.n)

    def act(self, state):
        """
        Act according to the behaviour_policy (b(a|s)), which is epsilon-greedy with respect to the q-values.
        """
        # N.B., break ties consistency with "last".
        # C.f. EpsilonGreedyPolicy.select_action and QValueTable.get_max_action for more info
        return self.behaviour_policy.select_action(state, self.q_values, ties="last")

    def _update_q_and_pi(self, episode):
        """Update q-values using first-visit MC"""

        # Initialise returns ("G") and weights ("W") for this episode
        returns = 0
        weights = 1

        # Starting from terminal state, work backwards through the episode
        for state, action, reward in reversed(episode):

            # G <- gamma * G + R_t+1
            returns = self.gamma * returns + reward

            # # TODO: check first-visit or every-visit?
            # if self._is_subelement_present((state, action), episode[:len(episode) - t - 1]):
            #     continue

            # C(S_t, A_t) <- C(S_t, A_t) + W
            self.state_action_counts.update_importance_sampling(state, action, weights)    # Update C(S_t, A_t)

            # Q(S_t, A_t) <- Q(S_t, A_t) + W / C(S_t, A_t) * (G - Q(S_t, A_t))
            step_size = weights / self.state_action_counts.get(state, action)    # W / C(S_t, A_t)
            new_value = self.q_values.get(state, action) + step_size * (returns - self.q_values.get(state, action))    # Q(S_t, A_t) + W / C(S_t, A_t) * (G - Q(S_t, A_t))
            self.q_values.update(state, action, new_value)    # Update Q(S_t, A_t)

            # Update the target policy (self.policy).
            # N.B. 1. Target policy `policy` is deterministic.
            #      2. Ties are broken consistently with "last".
            self.policy.update(state, self.q_values, ties="last")

            # If chosen action is not the same as the target policy, break (inner) loop
            if action != self.policy.select_action(state):
                break

            # W <- W * 1 / b(A_t | S_t)
            weights *= 1 / self.behaviour_policy.compute_probs(state, self.q_values)[action]

    def learn(self, num_episodes=10000):

        for episode in range(num_episodes):

            # Print progress
            if episode % 1000 == 0:
                print(f"Episode {episode}/{num_episodes}")

            episode = self._generate_episode()
            self._update_q_and_pi(episode)


def run():

    # Run parameters
    train_episodes = 50000

    # Create the environment
    env = gym.make("Blackjack-v1", sab=True)  # `sab` means rules following Sutton and Barto
    rl_loop = MCOffPolicy(env, epsilon=0.1, gamma=1.0)
    rl_loop.learn(num_episodes=train_episodes)

    # Plot the results
    plot_results(rl_loop)


if __name__ == "__main__":
    run()
