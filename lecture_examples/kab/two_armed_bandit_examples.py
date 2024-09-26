import pandas as pd


# # greedy
action_rewards = [(0, 3), (1, 0.5), (0, 0.4), (0, 1), (0, -1), (0, -0.5), (0, 0.5), (0, 0), (1, 0.3)]
# epsilon-greedy
# action_rewards = [(0, 3), (1, 0.5), (0, 0.4), (0, 1), (1, 2), (0, -0.5), (1, 1), (1, 1.5), (1, 0.5)]

# Weighted average (greedy)
# 0 initialisation
action_rewards = [(0, 3), (1, 0.5), (0, 0.4), (0, 1), (0, -1), (1, -0.5), (0, 0), (0, -0.5), (1, 0.3)]

# 0 initialisation
action_rewards_optimistic = [(0, 3), (1, 0.5), (0, 0.4), (1, 1), (0, -1), (1, -0.5), (1, 0.7), (1, 0.5), (1, 1.2)]

def simple_average(action_rewards, initial_value=0):

    times_selected = {
        0: 0,
        1: 0,
    }

    value_estimates = {
        0: initial_value,
        1: initial_value,
    }

    # Initialise an object which will store value estimates as time progresses
    # Keys are the timestep, values are the inner dictionary of value estimates
    value_estimates_over_time = {0: value_estimates.copy()}

    for step, (action, reward) in enumerate(action_rewards):
        # Value estimate is the average of all rewards seen so far, for each action.

        # Incremental update rule for average:
        # NewEstimate <- OldEstimate + 1/N * (Reward - OldEstimate)

        # Increment the number of times the action has been selected
        times_selected[action] += 1

        # Update the value estimate for the action
        value_estimates[action] += (1 / times_selected[action]) * (reward - value_estimates[action])

        # Store the value estimates at this timestep
        value_estimates_over_time[step + 1] = value_estimates.copy()

    # Get output into pandas table format
    value_estimates_over_time = pd.DataFrame(value_estimates_over_time)

    return value_estimates_over_time


def weighted_average(action_rewards, initial_value=0, alpha=0.1):

    times_selected = {
        0: 0,
        1: 0,
    }

    value_estimates = {
        0: initial_value,
        1: initial_value,
    }

    # Initialise an object which will store value estimates as time progresses
    # Keys are the timestep, values are the inner dictionary of value estimates
    value_estimates_over_time = {0: value_estimates.copy()}

    for step, (action, reward) in enumerate(action_rewards):
        # Value estimate is the average of all rewards seen so far, for each action.

        # Incremental update rule for average:
        # NewEstimate <- OldEstimate + 1/N * (Reward - OldEstimate)

        # Update the value estimate for the action
        value_estimates[action] += alpha * (reward - value_estimates[action])

        # Store the value estimates at this timestep
        value_estimates_over_time[step + 1] = value_estimates.copy()

    # Get output into pandas table format
    value_estimates_over_time = pd.DataFrame(value_estimates_over_time)

    return value_estimates_over_time


if __name__ == '__main__':
    # simple_value_estimates = simple_average(action_rewards)
    weighted_value_estimates = weighted_average(action_rewards, alpha=0.5)
    weighted_value_estimates_optimistic = weighted_average(action_rewards_optimistic, alpha=0.5, initial_value=5)

    # print(f"Simple average value estimates: {simple_value_estimates}")
    print(f"Weighted average value estimates, 0 initialisation: {weighted_value_estimates}")
    print(f"Weighted average value estimates, 5 initialisation: {weighted_value_estimates_optimistic}")
