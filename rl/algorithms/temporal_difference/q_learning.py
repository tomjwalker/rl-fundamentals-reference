from rl.algorithms.common.td_agent import TemporalDifferenceAgent


class QLearning(TemporalDifferenceAgent):

    def __init__(self, env, alpha=0.5, gamma=1.0, epsilon=0.1, logger=None, random_seed=None):
        super().__init__(env, gamma, alpha, epsilon, logger, random_seed)
        self.name = "Q-Learning"

    def learn(self, num_episodes: int = 500) -> None:

        for episode in range(num_episodes):

            # Initialise S
            state, _ = self.env.reset()

            # Loop over each step of episode, until S is terminal
            done = False
            while not done:

                # Choose A from S using policy derived from Q (epsilon-greedy)
                action = self.act(state)

                # Take action A, observe R, S'
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Update Q(S, A), taking as target the q-learning TD target (R + gamma * max_a Q(S', a))
                td_target = reward + self.gamma * self.q_values.get(next_state, self.q_values.get_max_action(next_state))
                td_error = td_target - self.q_values.get(state, action)
                new_value = self.q_values.get(state, action) + self.alpha * td_error
                self.q_values.update(state, action, new_value)

                # S <- S', A <- A'
                state = next_state

                # Add reward to episode reward
                self.logger.log_timestep(reward)

                # If S is terminal, then episode is done (exit loop)
                done = terminated or truncated

            # Add episode reward to list
            self.logger.log_episode()
