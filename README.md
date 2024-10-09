<p>
    <img alt="Logo" height="50" src="./images/general/logo_dark.png" style="vertical-align:middle; margin-right: 
10px;" id="logo">
    <span style="font-size:2em; font-weight:bold;">RL Fundamentals</span>
</p>

<style>
  @media (prefers-color-scheme: light) {
    #logo {
      content: url('./images/general/logo_light.png');
    }
  }
</style>

Welcome to the RL Fundamentals course! This repository contains all the materials, assignments, and examples you'll 
need to master Reinforcement Learning concepts.

## Repository Structure

- **assignments/**
  - Bandits: [bandits.md](assignments/bandits.md)
  - MDPs: [mdps.md](assignments/mdps.md)
  - Dynamic Programming: [dynamic_programming.md](assignments/dynamic_programming.md)
  - Monte Carlo: [monte_carlo.md](assignments/monte_carlo.md)
  - Temporal Difference: [temporal_difference.md](assignments/temporal_difference.md)
  - Planning: [planning.md](assignments/planning.md)
- **rl/**: Contains the core reinforcement learning algorithms and environment implementations.
- **exercises/**: Additional exercises to reinforce learning.
- **lecture_examples/**: Examples used in lectures to illustrate key concepts.
- **utils/**: Utility scripts and helper functions.

## Getting Started

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/rl-fundamentals-course.git
    ```

2. **Navigate to the Repository:**

    ```bash
    cd rl-fundamentals-course
    ```

3. **Prepare the Student Repository:**

    Use the provided `prepare_student_repo.py` script to generate a student-friendly version of the repository.

    ```bash
    python utils/prepare_student_repo.py . ../rl-fundamentals-assignments --dirs rl exercises
    ```

4. **Explore Assignments:**

    Each assignment has its own markdown file within the `assignments/` directory. Navigate to the desired assignment folder and follow the guidelines to complete the tasks.

## Submitting Assignments

Ensure that you follow the submission guidelines provided in each assignment's markdown file. Typically, you'll need to submit your modified Python files along with any required reports or observations.

## Additional Resources

- **Books:**
  - Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (Second Edition).

- **Documentation:**
  - [NumPy Documentation](https://numpy.org/doc/)
  - [Pandas Documentation](https://pandas.pydata.org/docs/)
  - [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

## Contact

For any questions or issues, please contact [Your Name](mailto:your.email@example.com).
