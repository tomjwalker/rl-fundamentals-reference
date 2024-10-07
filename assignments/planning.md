# Assignment: Planning

## Overview

In this assignment, you will implement Planning methods for solving reinforcement learning problems with greater 
sample efficiency.

Algorithms:
- Dyna-Q
- Dyna-Q+

Environments:
- Blocking Maze
- Shortcut Maze

## Objectives

- [Objective 1]
- [Objective 2]
- [Objective 3]

## Files to Work On

- `rl/algorithms/planning/dyna.py`
- `rl/algorithms/planning/dyna_plus.py`

---

## Instructions

### Step 1: Dyna-Q

1. **Complete the `DynaModel` helper class in `dyna.py`:**
   - An instance of this class stores the model of the environment in the main Dyna-Q algorithm.
   - Complete the `add` method to store the transition in the model.
   - Complete the `sample_state_action` method to sample a state-action pair from the model.
2. **Complete the main `DynaQ.learn` method in `dyna.py`:**
   - (`__init__` and `act` methods are already implemented, but inspect them for understanding.)
   - The learn method is left blank (less hand-holding than previous assignments) for you to complete. 
   - Remember, most of the algorithm closely follows Q-learning
3. **Run `[Experiment/Script]`:**
   - TODO XXXXXX
   - Execute the script:
     ```bash
     xxx
     ```

### Step 2: Dyna-Q+

1. **Complete the `DynaPlusModel` helper class in `dyna_plus.py`:**
    - Note, this inherits from `DynaModel`, so most methods already implemented.
    - A new `add` method is required to initialise all possible actions for an encountered state.
    - Complete the `add` method to store the transition in the model.
    - Complete the `sample_state_action` method to sample a state-action pair from the model.
2. **Complete the `TimeSinceLastEncountered` helper class in `dyna_plus.py`:**
    - This implements the tau(s, a) object tracking the time since the last encounter of a state-action pair.
    - (Inherits from `QValueTable`, essentially an S x A NumPy array. Has `get`, `update` methods already implemented.)
    - Complete the `increment` method, which increments non-encountered (s, a) pairs and resets encountered pairs.
3.  **Complete the main `DynaQPlus.learn` method in `dyna_plus.py`:**
    - (`__init__` and `act` methods are already implemented, but inspect them for understanding.)
    - The learn method is left blank (less hand-holding than previous assignments) for you to complete. 
    - Note that `DynaPlus` inherits from the `Dyna` class
    - Note additional attributes for this class: `self.kappa`, `self.time_since_last_encountered`, and the altered `self.model` attribute.
3. **Run `[Experiment/Script]`:**
   - Execute the script for the Blocking Maze:
     ```bash
     python -m rl.experiments.planning.non_stationary_environments
     ```
    - Execute the script for the Shortcut Maze:
      ```bash
      python -m rl.experiments.planning.non_stationary_environments --environment_name ShortcutMaze
      ```
      - N.B. you can run one rather than both algorithm with `--agents Dyna-Q` etc
      - N.B. you can adjust other trial parameters, e.g. `--train_episodes 50 --n_runs 2`
   

## Expected Outputs

### [Experiment Name]

![Description of Image](images/[assignment_name]/[image_name].png)

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
