# Assignment: Epsilon-Greedy Algorithm and Random Tie-Breaking

## Overview

In this assignment, you will implement the epsilon-greedy algorithm for the k-armed bandit problem and the `argmax_ties_random` function. This exercise will help you understand the exploration-exploitation trade-off and the importance of handling ties in action selection.

## Objectives

- Implement the epsilon-greedy action selection strategy.
- Implement the `argmax_ties_random` function to handle ties randomly.
- Understand the impact of different epsilon values and initialization strategies on the agent's performance.
- Run experiments to observe and analyze the agent's behavior under various settings.

## Files to Work On

- `rl/utils/general.py`
  - **Function to Implement:**
    - `argmax_ties_random`
- `rl/algorithms/bandits/epsilon_greedy.py`
  - **Methods to Complete:**
    - `reset`
    - `act`
    - `simple_update`
    - `weighted_update`

## Instructions

### 1. Implement `argmax_ties_random` in `general.py`

Navigate to `rl/utils/general.py` and locate the `argmax_ties_random` function marked with `# HOMEWORK`. Implement the function as per the instructions.

**Hints:**

- Use `np.max` to find the maximum value in `q_values`.
- Use `np.where` to find indices of elements equal to the maximum value.
- Use `np.random.choice` to randomly select one of the indices if there are ties.

### 2. Complete the `EpsilonGreedy` Class in `epsilon_greedy.py`

In `rl/algorithms/bandits/epsilon_greedy.py`, complete the methods marked with `# HOMEWORK`.

#### a. `reset` Method

Implement the logic to initialize the action-value estimates (`self.q_values`) and action counts (`self.action_counts`) based on the `self.initialisation` value.

#### b. `act` Method

Implement the epsilon-greedy action selection:

- With probability `epsilon`, select a random action.
- With probability `1 - epsilon`, select the action with the highest estimated value using `argmax_ties_random`.

#### c. `simple_update` Method

Update the action-value estimate using sample averages.

**Formula:**

Q_{n+1} = Q_n + (1 / N_n) * (R_n - Q_n)

arduino
Copy code

#### d. `weighted_update` Method

Update the action-value estimate using a constant step size (`alpha`).

**Formula:**

Q_{n+1} = Q_n + α * (R_n - Q_n)

markdown
Copy code

## Running Experiments

After completing the implementations, you can run the experiments to see the agent's performance.

### a. Epsilon Sweep Experiment

1. Open `epsilon_greedy.py`.
2. In the `if __name__ == "__main__":` block, uncomment the following line to run the epsilon sweep experiment:

    ```python
    # Epsilon Sweep Experiment
    epsilon_sweep_experiment()
    ```

3. Run the script:

    ```bash
    python rl/algorithms/bandits/epsilon_greedy.py
    ```

4. Observe the plotted results showing average rewards and optimal action percentages over time for different epsilon values.

### b. Optimistic Initial Values Experiment

1. Open `epsilon_greedy.py`.
2. In the `if __name__ == "__main__":` block, uncomment one of the following lines to run the experiment with desired options:

    ```python
    # Run with default settings
    initial_val_experiment()

    # Run showing individual runs
    # initial_val_experiment(show_individual_runs=True)

    # Run showing confidence intervals
    # initial_val_experiment(show_confidence_interval=True)

    # Run showing individual runs and confidence intervals
    # initial_val_experiment(show_individual_runs=True, show_confidence_interval=True)
    ```

3. Run the script:

    ```bash
    python rl/algorithms/bandits/epsilon_greedy.py
    ```

4. Observe the plotted results demonstrating the effect of different initialization strategies on the agent's performance.

## Submission

- Ensure that all methods and functions are correctly implemented.
- Include any observations or insights you gained from the experiments in a separate report or comment block.
- Submit the modified `general.py` and `epsilon_greedy.py` files.

## Additional Resources

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (Second Edition). Chapters 2.
- NumPy Documentation: [np.max](https://numpy.org/doc/stable/reference/generated/numpy.amax.html), [np.where](https://numpy.org/doc/stable/reference/generated/numpy.where.html), [np.random.choice](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html)

## Notes

- Make sure to set the random seed as provided to ensure reproducible results.
- Pay attention to the use of vectorized operations with NumPy for efficiency.
- If you encounter any issues, double-check your implementations and consult the resources provided