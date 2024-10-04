# Assignment: [Assignment Title]

## Overview

[Provide a brief overview of the assignment and its significance.]

## Objectives

- [Objective 1]
- [Objective 2]
- [Objective 3]

## Files to Work On

- `[path/to/file1.py]`
  - **Function to Implement:**
    - `[function_name]`
- `[path/to/file2.py]`
  - **Method to Complete:**
    - `[method_name]`

---

## Moving to Object-Oriented Programming (OOP)

TODO: Discuss motivations for moving to OOP (c.f. Bandit / Dynamic Programming repeated code).

## Instructions

<details>
<summary><h3>Step 1: Exploring Starts</h3></summary>

0. **Inspect**: You will be focussing on the `act` and `learn` methods, but inspect other methods in the 
`MCExploringStartsAgent`, along with the objects it inherits from and interacts with.
    - N.B., policy is initialised with one that sticks (0) for hands of 20 or 21, and hits (1) otherwise.
    - This is encoding environment-specific knowledge, which is specific to this Blackjack assignment.
1. **Implement `MCExploringStartsAgent.act` in `exploring_starts.py`:**
   - This is a single line of code in the method itself
   - However, it will require you to also implement `select_action` within the `DeterministicPolicy` class in `policy.py`
2. **Implement `MCExploringStartsAgent.learn` in `exploring_starts.py`:**
   - This will include completing the helper method in the superclass, `MonteCarloAgent._generate_episode`
   - Useful attributes: `self.state_action_counts`, `self.q_values`, `self.policy`
3. **Run:**
   - Execute the script (from the root directory in the terminal):
     ```bash
     python -m rl.algorithms.monte_carlo.exploring_starts
     ```
   - If you want more converged results closer to those in the lecture slides, 
   you can increase the number of episodes to 500,000.
     ```bash
     python -m rl.algorithms.monte_carlo.exploring_starts --num_episodes 500000
     ```
3. **Observe:** 

![Exploring starts](../images/monte_carlo/exploring_starts.png)

*A learnt policy after 500,000 iterations. Top plots: state value function. Bottom plots: policies*
</details>

<details>
<summary><h3>Step 2: On-policy control</h3></summary>

0. **Inspect**: You will be focussing on the `act` and `learn` methods, but inspect other methods in the 
`MCExploringStartsAgent`, along with the objects it inherits from and interacts with.
    - Initialised with an `EpisilonGreedyPolicy` policy object.
    - N.B., unlike the Exploring Starts demo, here policy is initialised as all zeros (more general approach - no prior 
   knowledge).
1. **Implement `MCExploringStartsAgent.act` in `exploring_starts.py`:**
   - This is a single line of code in the method itself
   - However, it will require you to also implement `select_action` within the `EpsilonGreedyPolicy` class in 
     `policy.py` 
   - This is around 6 lines of code (less handholding at points from now on, but you can refer to a similar 
     implementation from the multi-armed bandit assignment)
2. **Implement `MCExploringStartsAgent.learn` in `exploring_starts.py`:**
   - This is almost identical to the learn method of Exploring Starts - most of the different behaviour comes from the 
     policy object and its use generating episodes.
3. **Run:**
   - Execute the script (from the root directory in the terminal):
     ```bash
     python -m rl.algorithms.monte_carlo.on_policy
     ```
   - If you want more converged results closer to those in the lecture slides, 
   you can increase the number of episodes to 500,000.
     ```bash
     python -m rl.algorithms.monte_carlo.on_policy --num_episodes 500000
     ```
3. **Observe:** 

![Exploring starts](../images/monte_carlo/on_policy.png)

*A learnt policy after 500,000 iterations. Top plots: state value function. Bottom plots: policies*

</details>

<details>
<summary><h3>Step 3: Off-policy control</h3></summary>

1. **Implement `[Function/Method]` in `[File]`:**
   - [Brief instruction or description.]
   - **Hint:** [Optional hint.]

2. **Run `[Experiment/Script]`:**
   1. Open `[file_path]`.
   2. Execute the script:
     ```bash
     python [path/to/script.py]
     ```
3. **Observe:** [Brief description of expected observations.]

</details>


*Brief description of what the image illustrates.*

---

## Additional Resources

- [Resource 1](URL)
- [Resource 2](URL)

## Notes

- Ensure reproducibility by setting the random seed.
- Utilize vectorized operations for efficiency.
- Verify dependencies: NumPy, Pandas, Matplotlib, etc.
- Experiment with different parameters to enhance understanding.

---
Good luck with your assignment!
