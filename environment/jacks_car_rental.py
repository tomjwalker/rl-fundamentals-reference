import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def get_poisson_prob(x, lam, high):
    """
    - if X > high, then probability is as per a Poisson distribution, p(X|lam)
    - if X = high, then the probability is the poisson probability for P(X=high|lam) + the sum of all further
        probabilities in the tail, to infinity, as P(X > high) = 0, per the constraints of the problem
    """

    if x < high:
        return np.exp(-lam) * (lam ** x) / math.factorial(x)
    elif x == high:
        # Get sum of all probabilities up to, but excluding, high
        prob = 0
        for i in range(x):
            prob += np.exp(-lam) * (lam ** i) / math.factorial(i)
        # Probability of tail, including and beyond high is 1 - prob
        return 1 - prob
    else:
        return 0


class Location:

    def __init__(self, mean_requests, mean_returns):
        self.mean_requests = mean_requests
        self.mean_returns = mean_returns


class JacksCarRental:

    def __init__(self, max_cars=20, max_move_cars=5, rental_reward=10, move_cost=2, random_seed=None):
        self.max_cars = max_cars
        self.max_move_cars = max_move_cars
        self.rental_reward = rental_reward
        self.move_cost = move_cost
        self.random_seed = random_seed
        self.locations = self._init_locations()

    @staticmethod
    def _init_locations():
        """
        Initialise the rental locations.
        """
        return [
            Location(mean_requests=3, mean_returns=3),
            Location(mean_requests=4, mean_returns=2),
        ]

    def get_expected_reward(self, state, action):
        """
        Get the expected reward for a given state and action.

        The state s = (n1, n2) is the number of cars at each location at the end of the day.
        The expected reward is a function of the number of cars available the following day, in the following way:
        - cars available at the start of the day, after overnight distribution:
            n1'' = n1 - a
            n2'' = n2 + a
        - The expected reward is then:
        r(s, a) = rental_reward * sum_over_possible_X1s(X1 * p(X1|lam, N1''))
            + sum_over_possible_X2s(X2 * p(X2|lam, N2''))
            - move_cost * abs(a)

        Args:
            state (tuple): The state: number of cars at end of day: (location 1, location 2).
            action (int): The action, ranging from -5 to 5 (+ve = move cars from location 1 to location 2, -ve = move
                cars from location 2 to location 1).

        Returns:
            float: The expected reward.
        """
        expected_reward = 0
        for location_idx, location in enumerate(self.locations):
            # Get the number of cars available at the start of the day, after overnight distribution (n'')
            if location_idx == 0:
                cars_available = state[0] - action
            else:
                cars_available = state[1] + action
            # assert cars_available is of type int
            # Get the expected reward for this location: sum over all possible Xs (number of rentals)
            for x in range(cars_available + 1):
                expected_reward += self.rental_reward * x * get_poisson_prob(x, location.mean_requests, cars_available)
        # Subtract the cost of moving cars
        expected_reward -= self.move_cost * abs(action)
        return expected_reward

    def get_state_transition_probs(self, next_state, state, action):
        """
        Get the probability of transitioning from state to next_state, given action.

        Calculated the following way:
        - cars available at the start of the day, after overnight distribution:
            n1'' = n1 - a
            n2'' = n2 + a
        - Let the next state s' = (n1', n2') be the number of cars at each location at the end of the day.
        - Let the Poisson distributed random variables X1 and X2 be the number of rentals at each location.
        - Let the Poisson distributed random variables Y1 and Y2 be the number of returns at each location.
        - So that the number of cars at each location at the end of the day is:
            N1' = n1'' + Y1 - X1
            N2' = n2'' + Y2 - X2
        - Then the probability of transitioning from state to next_state is:
            p(s'|s, a) = p(N1', N2'|n1'', n2'')
        - With independence assumptions, this is:
            p(s'|s, a) = p(N1'|N1'') * p(N2'|N2'')
        - The probability of starting the day with N1'' cars and finishing with N1' cars is:
            p(N1'|N1'') = sum_over_possible_X1s(p(X1|lam_rent, N1'') * p(Y1 = N1' - N1'' + X1|lam_return,
            N1_max - N1'' + X1))
        - (The same applies for N2)
        """
        # Get the number of cars available at the start of the day, s'' = (n1'', n2'')
        state_start_day = (state[0] - action, state[1] + action)
        # Get the number of cars at the end of the day, s' = (n1', n2')
        state_end_day = next_state
        # Get the probability of transitioning from state to next_state
        prob = 1
        for location_idx, location in enumerate(self.locations):
            # Get the number of cars at the start of the day, after overnight distribution (n'')
            cars_available = state_start_day[location_idx]
            # Get the number of cars at the end of the day (n')
            cars_end_day = state_end_day[location_idx]
            # Get the probability of transitioning from n'' to n'
            prob_for_location = 0
            for x in range(cars_available + 1):
                prob_x = get_poisson_prob(x, location.mean_requests, cars_available)
                num_returns = cars_end_day - cars_available + x
                if num_returns < 0:
                    # If there are more rentals than cars available, then the probability is 0
                    prob_y = 0
                else:
                    prob_y = get_poisson_prob((cars_end_day - cars_available + x), location.mean_returns,
                                              (self.max_cars - cars_available + x))
                prob_for_location += prob_x * prob_y
            prob *= prob_for_location
        return prob


def main():
    # ## Testing truncated Poisson distribution
    # # With a lambda of 3, and an X high of 10, plot the pmf for X = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    # # and show that the sum of all probabilities is 1
    # lam = 8
    # high = 10
    # probs = []
    # for x in range(high + 1):
    #     probs.append(get_poisson_prob(x, lam, high))
    # # Plot the pmf
    #
    # plt.bar(range(high + 1), probs)
    # plt.title(f"Poisson distribution with lambda={lam}. Sum of probabilities is {sum(probs)}")
    # plt.show()
    # # Show that the sum of all probabilities is 1
    # print(sum(probs))

    ## Testing Jack's Car Rental environment
    # Create the environment
    env = JacksCarRental()
    # Get the expected reward for a given state and action
    state = (10, 10)
    action = 3
    expected_reward = env.get_expected_reward(state, action)
    print(f"Expected reward for state {state} and action {action} is {expected_reward}")
    # Get the probability of transitioning from state to next_state, given action
    next_state = (5, 15)
    prob = env.get_state_transition_probs(next_state, state, action)
    print(f"Probability of transitioning from state {state} to state {next_state} given action {action} is {prob}")


if __name__ == "__main__":
    main()
