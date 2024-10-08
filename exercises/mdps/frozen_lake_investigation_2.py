import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import sys

# Use TkAgg backend for matplotlib
matplotlib.use('TkAgg')

# Discount factor
GAMMA = 0.5

# Predefined maps
MAP_TWO_TWO = ["SF", "FG"]
MAP_THREE_THREE = ["SFF", "HFH", "FFG"]

# Define actions
ACTIONS = ['U', 'D', 'L', 'R']  # Up, Down, Left, Right


def parse_map(desc):
    """
    Parses the map description and assigns state indices.

    Returns:
        grid: 2D list of characters
        state_indices: dict mapping (row, col) to state index
        terminal_states: dict mapping state index to reward upon entering
    """
    grid = [list(row) for row in desc]
    rows = len(grid)
    cols = len(grid[0])
    state_indices = {}
    terminal_states = {}
    state = 0
    for r in range(rows):
        for c in range(cols):
            state_indices[(r, c)] = state
            if grid[r][c] == 'H':
                terminal_states[state] = 0  # Hole has reward 0 upon entering
            elif grid[r][c] == 'G':
                terminal_states[state] = 1  # Goal has reward 1 upon entering
            state += 1
    return grid, state_indices, terminal_states


def get_next_state(r, c, action, rows, cols):
    """
    Given a position and action, returns the next position after taking the action.

    If the action leads off the grid, the agent stays in the same position.
    """
    if action == 'U':
        new_r = max(r - 1, 0)
        new_c = c
    elif action == 'D':
        new_r = min(r + 1, rows - 1)
        new_c = c
    elif action == 'L':
        new_r = r
        new_c = max(c - 1, 0)
    elif action == 'R':
        new_r = r
        new_c = min(c + 1, cols - 1)
    else:
        raise ValueError("Invalid action")
    return new_r, new_c


def solve_frozenlake(desc, gamma=GAMMA, policy=None):
    """
    Solves the FrozenLake environment analytically for state values.

    Args:
        desc: list of strings representing the map
        gamma: discount factor
        policy: dict mapping state to action probabilities. If None, assumes equiprobable.

    Returns:
        v: numpy array of state values or None if unsolvable
    """
    grid, state_indices, terminal_states = parse_map(desc)
    rows = len(grid)
    cols = len(grid[0])
    num_states = rows * cols

    if policy is None:
        # Equiprobable policy: each action has probability 1/4
        policy = {s: {a: 1/4 for a in ACTIONS} for s in range(num_states) if s not in terminal_states}

    # Initialize A and b for the linear equations Av = b
    A = np.zeros((num_states, num_states))
    b = np.zeros(num_states)

    for r in range(rows):
        for c in range(cols):
            s = state_indices[(r, c)]
            if s in terminal_states:
                # For terminal states: v(s) = 0
                A[s, s] = 1
                b[s] = 0
            else:
                # For non-terminal states: v(s) = sum_a pi(a|s) [ R(s,a) + gamma * v(s') ]
                A[s, s] = 1  # Coefficient for v(s)
                for a in ACTIONS:
                    a_prob = policy[s][a]
                    next_r, next_c = get_next_state(r, c, a, rows, cols)
                    s_prime = state_indices[(next_r, next_c)]
                    if s_prime in terminal_states:
                        R = terminal_states[s_prime]  # Reward upon entering terminal state
                    else:
                        R = 0  # No immediate reward
                    # Add to the right-hand side
                    b[s] += a_prob * R
                    # Subtract gamma * pi(a|s) * v(s') from the left-hand side
                    A[s, s_prime] -= gamma * a_prob

    try:
        # Solve Av = b
        v = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        print(f"Linear system could not be solved: {e}")
        v = None

    return v


def display_values(v, desc):
    """
    Displays the state values in a grid format.

    Args:
        v: numpy array of state values
        desc: list of strings representing the map
    """
    if v is None:
        print("No values to display due to previous errors.\n")
        return

    grid = [list(row) for row in desc]
    rows = len(grid)
    cols = len(grid[0])
    print("State Values:")
    for r in range(rows):
        row_vals = ""
        for c in range(cols):
            s = r * cols + c
            row_vals += f"{v[s]:.3f} "
        print(row_vals)
    print("\n")


def perform_sweep(start_size=2, end_size=92, step=10, runs_per_size=5):
    """
    Performs a sweep over map sizes from start_size to end_size with specified steps.

    For each size, generates `runs_per_size` random maps, solves them,
    measures the solution time, and averages the results.

    Args:
        start_size: Starting side length (inclusive)
        end_size: Ending side length (inclusive)
        step: Step size between consecutive side lengths
        runs_per_size: Number of runs per map size

    Returns:
        sizes: List of map sizes
        avg_times: List of average solution times corresponding to each size
        std_times: List of standard deviations of solution times corresponding to each size
    """
    sizes = list(range(start_size, end_size + 1, step))
    avg_times = []
    std_times = []

    for size in sizes:
        times = []
        print(f"Processing map size: {size}x{size}")
        for run in range(1, runs_per_size + 1):
            # Generate a random map
            random_map = generate_random_map(size=size)
            # Ensure the map has exactly one start 'S' and one goal 'G'
            # Gym's generate_random_map ensures this
            try:
                start_time = time.time()
                v = solve_frozenlake(random_map, gamma=GAMMA)
                end_time = time.time()
                elapsed_time = end_time - start_time
                if v is not None:
                    times.append(elapsed_time)
                    print(f"  Run {run}: Time = {elapsed_time:.6f} seconds")
                else:
                    print(f"  Run {run}: Failed to solve the map.")
            except Exception as e:
                # Handle unexpected errors
                print(f"  Run {run}: Encountered an error: {e}. Skipping this run.")
        if times:
            average_time = sum(times) / len(times)
            std_time = np.std(times)
            avg_times.append(average_time)
            std_times.append(std_time)
            print(f"  Average Time for size {size}x{size}: {average_time:.6f} seconds")
            print(f"  Standard Deviation for size {size}x{size}: {std_time:.6f} seconds\n")
        else:
            avg_times.append(None)
            std_times.append(None)
            print(f"  No valid runs for size {size}x{size}.\n")

    return sizes, avg_times, std_times


def plot_sweep(sizes, avg_times, std_times):
    """
    Plots the sweep results: map size vs average solution time, with error bars.

    Args:
        sizes: List of map sizes
        avg_times: List of average solution times
        std_times: List of standard deviations of solution times
    """
    # Filter out sizes with None average times
    filtered_data = [(s, t, std) for s, t, std in zip(sizes, avg_times, std_times) if t is not None]
    if not filtered_data:
        print("No valid data to plot.")
        return

    filtered_sizes, filtered_avg_times, filtered_std_times = zip(*filtered_data)

    plt.figure(figsize=(12, 8))
    plt.errorbar(filtered_sizes, filtered_avg_times, yerr=filtered_std_times, fmt='o-',
                 elinewidth=3, capsize=0, label='Average Solution Time')
    plt.title('FrozenLake Analytical Solution Time vs Map Size (5 runs per size)')
    plt.xlabel('Map Side Length (n)')
    plt.ylabel('Average Solution Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.xticks(filtered_sizes)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="FrozenLake Analytical Investigations")
    subparsers = parser.add_subparsers(dest='mode', help='Modes of operation')

    # Subparser for 'solve' mode
    parser_solve = subparsers.add_parser('solve', help='Solve specific 2x2 and 3x3 FrozenLake maps')

    # Subparser for 'sweep' mode
    parser_sweep = subparsers.add_parser('sweep', help='Perform a sweep over different map sizes')
    parser_sweep.add_argument('--start_size', type=int, default=2, help='Starting side length (default: 2)')
    parser_sweep.add_argument('--end_size', type=int, default=92, help='Ending side length (default: 92)')
    parser_sweep.add_argument('--step', type=int, default=10, help='Step size between side lengths (default: 10)')
    parser_sweep.add_argument('--runs', type=int, default=5, help='Number of runs per map size (default: 5)')

    args = parser.parse_args()

    if args.mode == 'solve':
        # Solve and display the two by two grid
        print("Solving 2x2 Frozen Lake:")
        v_two = solve_frozenlake(MAP_TWO_TWO, gamma=GAMMA)
        display_values(v_two, MAP_TWO_TWO)

        # Solve and display the three by three grid
        print("Solving 3x3 Frozen Lake:")
        v_three = solve_frozenlake(MAP_THREE_THREE, gamma=GAMMA)
        display_values(v_three, MAP_THREE_THREE)

    elif args.mode == 'sweep':
        # Perform the sweep with user-specified parameters
        print(f"Performing sweep over map sizes from {args.start_size}x{args.start_size} to {args.end_size}x{args.end_size} with step {args.step}...")
        sizes, avg_times, std_times = perform_sweep(
            start_size=args.start_size,
            end_size=args.end_size,
            step=args.step,
            runs_per_size=args.runs
        )

        # Plot the sweep results
        print("Plotting the sweep results...")
        plot_sweep(sizes, avg_times, std_times)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
