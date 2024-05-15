"""
TODO:
    - Figure out strange target_policy plots
"""

from algorithms.monte_carlo.viz import plot_results
# N.B., numpy argmax used in this instance to ensure ties are broken consistently

import gymnasium as gym
import numpy as np

import matplotlib
matplotlib.use('TkAgg')


# TODO: move this to utils (as with other MC modules)
def _is_subelement_present(subelement, my_list):
    """
    Helps check if a subelement is present in a list of tuples. Used to check if state has already been seen.

    Simple example:
    _is_subelement_present((1, 2), [(1, 2, 3), (4, 5, 6)])
        True
    """
    for tpl in my_list:
        if subelement == tpl[:len(subelement)]:
            return True
    return False


class MCOffPolicy:

    def __init__(self, env, epsilon=0.1, gamma=1.0):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma

        self.name = "MC Off-Policy"    # For plotting

        self.q_values = None
        self.target_policy = None
        self.returns = None
        self.weights_cumsum = None
        self.behaviour_policy = None
        self.reset()

    def reset(self):
        # Get env shape
        state_shape = ()
        for space in self.env.observation_space:
            state_shape += (space.n,)

        # Initialise q-values
        state_and_action_shape = state_shape + (self.env.action_space.n,)
        self.q_values = np.zeros(state_and_action_shape)

        # TODO: check this
        # Target target_policy is deterministic (π(s)), and is initialised as argmax of q-values, along the action axis
        self.target_policy = np.argmax(self.q_values, axis=-1)    # np.argmax ensures ties broken consistently

        # Initialise weights cumsum ("C" in Sutton and Barto, pp. 109)
        self.weights_cumsum = np.zeros(state_and_action_shape)

        # Initialise behaviour policy (b(a|s)), which is epsilon-greedy with respect to the q-values
        self._update_behaviour_policy()

    def act(self, state):
        """
        Act according to the behaviour_policy (b(a|s)), which is epsilon-greedy with respect to the q-values.
        """
        # TODO: check no issues here around consistency of ties
        return np.random.choice(self.env.action_space.n, p=self.behaviour_policy[state])

    def _generate_episode(self):

        episode = []
        state, info = self.env.reset()
        done = False
        while not done:
            action = self.act(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated
        return episode

    def _update_q_and_pi(self, episode):
        """Update q-values using first-visit MC"""

        # Initialise returns ("G") and weights ("W") for this episode
        returns = 0
        weights = 1

        # Starting from terminal state, work backwards through the episode
        for state, action, reward in reversed(episode):
            returns = self.gamma * returns + reward

            # TODO: check first-visit or every-visit?
            # if not _is_subelement_present((state, action), episode[:-1]):
            #     self.returns[state][action].append(returns)
            #     self.q_values[state][action] = np.mean(self.returns[state][action])
            #     self._update_behaviour_policy(state)

            self.weights_cumsum[state][action] += weights
            update_ratio = weights / self.weights_cumsum[state][action]
            self.q_values[state][action] += update_ratio * (returns - self.q_values[state][action])
            self.target_policy[state] = np.argmax(self.q_values[state])    # np.argmax ensures ties broken consistently
            if self.target_policy[state] != action:
                break
            weights *= 1 / self.behaviour_policy[state][action]

    def _update_behaviour_policy(self):
        """
        Update the behaviour_policy, which acts epsilon-greedily with respect to the q-values.
        Where there are ties, these are broken consistently via np.argmax.
        b(a|s) is:
            - probability 1 - epsilon - epsilon / |A(s)| for the action with the highest q-value
            - epsilon /  |A(s)| for all other actions
        """

        random_prob = self.epsilon / self.env.action_space.n
        self.behaviour_policy = np.ones_like(self.q_values) * random_prob
        best_action = np.argmax(self.q_values, axis=-1)

        # TODO: cleaner way to do this?
        for state in np.ndindex(self.behaviour_policy.shape[:-1]):
            self.behaviour_policy[state][best_action[state]] = 1 - self.epsilon + random_prob

        # Assert that the behaviour_policy sums to 1 for all states
        assert np.allclose(np.sum(self.behaviour_policy, axis=-1), 1.0), \
            f"Policy does not sum to 1 for all states. Sum is {np.sum(self.behaviour_policy, axis=-1)}"

    def train(self, num_episodes=10000):

        for episode in range(num_episodes):

            # Print progress
            if episode % 1000 == 0:
                print(f"Episode {episode}/{num_episodes}")

            self._update_behaviour_policy()
            episode = self._generate_episode()
            self._update_q_and_pi(episode)


def run():

    # Run parameters
    train_episodes = 10000

    # Create the environment
    env = gym.make("Blackjack-v1", sab=True)  # `sab` means rules following Sutton and Barto
    rl_loop = MCOffPolicy(env)
    rl_loop.train(num_episodes=train_episodes)

    # Plot the results
    plot_results(rl_loop)


if __name__ == "__main__":
    run()
