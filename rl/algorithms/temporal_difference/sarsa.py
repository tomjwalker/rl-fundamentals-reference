from rl.algorithms.common.td_agent import TemporalDifferenceAgent


class Sarsa(TemporalDifferenceAgent):

    def __init__(self, env, alpha=0.5, gamma=1.0, epsilon=0.1, logger=None, random_seed=None):
        super().__init__(env, gamma, alpha, epsilon, logger, random_seed)
        self.name = "Sarsa"

    def learn(self, num_episodes: int = 500) -> None:

        for episode in range(num_episodes):

            # Initialise S
            state, _ = self.env.reset()

            # Choose A from S using policy derived from Q (epsilon-greedy)
            action = self.act(state)

            # Loop over each step of episode, until S is terminal
            done = False
            while not done:

                # Take action A, observe R, S'
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Choose A' from S' using policy derived from Q (epsilon-greedy)
                next_action = self.act(next_state)

                # Update Q(S, A), taking as target the TD target (R + gamma * Q(S', A'))
                td_target = reward + self.gamma * self.q_values.get(next_state, next_action)
                td_error = td_target - self.q_values.get(state, action)
                new_value = self.q_values.get(state, action) + self.alpha * td_error
                self.q_values.update(state, action, new_value)

                # S <- S', A <- A'
                state = next_state
                action = next_action

                # Add reward to episode reward
                self.logger.log_timestep(reward)

                # If S is terminal, then episode is done (exit loop)
                done = terminated or truncated

            # Add episode reward to list
            self.logger.log_episode()
