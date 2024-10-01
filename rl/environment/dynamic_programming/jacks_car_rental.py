
import numpy as np
from scipy.stats import poisson
import matplotlib
matplotlib.use('TkAgg')


def get_truncated_poisson_pmf(high, lam):
    """
    Using SciPy's poisson.pmf, get the probability mass function for a truncated Poisson distribution, from (0,
    high), both inclusive.
    p(X=high|lam) = 1 - sum_over_possible_Xs(p(X < high|lam)) to ensure that the sum of all probabilities is 1.

    Args:
        high (int): The upper bound of the truncated Poisson distribution.
        lam (float): The lambda of the Poisson distribution.

    Returns:
        list: A list of probabilities for each value in the truncated Poisson distribution.
    """
    # Get the pmf for a Poisson distribution with lambda=lam
    pmf = poisson.pmf(np.arange(high + 1), lam)
    # Get the sum of all probabilities up to, but excluding, high
    prob = 0
    for i in range(high):
        prob += pmf[i]
    # Probability of tail, including and beyond high is 1 - prob
    pmf[high] = 1 - prob
    return pmf


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

        self.locations = None
        self.expected_rental_reward = None
        self.prob_matrices = None
        self._init_locations()
        self._init_expected_rental_reward()
        self._init_transition_probs()

    def _init_locations(self):
        """
        Initialise the rental locations.
        """
        self.locations = [
            Location(mean_requests=3, mean_returns=3),
            Location(mean_requests=4, mean_returns=2),
        ]

    def _get_expected_reward_for_location(self, location_idx):
        """
        Returns the expected requests, sum_over_possible_Xs(X_i * p(X_i|lam_i)) for location i, where:
        - X_i = number of rentals at location i
        """

        # X_i = number of rentals at location i. As a vector, this is all possible values of X_i, from 0 to max_cars
        possible_rentals = np.arange(self.max_cars + 1)
        # p(X_i|lam_i) = probability of X_i rentals at location i
        # prob_rentals = poisson.pmf(possible_rentals, self.locations[location_idx].mean_requests)
        # # prob_rentals = get_truncated_poisson_pmf(high=self.max_cars, lam=self.locations[location_idx].mean_requests)

        expected_requests = np.zeros(len(possible_rentals))
        for possible_rentals_max in possible_rentals:    # Loop over all possible start of day car numbers
            # p(X_i|lam_i, N_i'') = probability of X_i rentals at location i, given N_i'' cars available at
            # start of day
            prob_to_max = get_truncated_poisson_pmf(
                high=possible_rentals_max,
                lam=self.locations[location_idx].mean_requests
            )
            # E[X_i|N_i''] = expected number of rentals at location i, given N_i'' cars available at start of day
            expected_requests[possible_rentals_max] = np.sum(possible_rentals[:possible_rentals_max + 1] * prob_to_max)

        return expected_requests

    def _init_expected_rental_reward(self):

        expected_requests_0 = self._get_expected_reward_for_location(0)
        expected_requests_1 = self._get_expected_reward_for_location(1)

        self.expected_rental_reward = self.rental_reward * (expected_requests_0[:, np.newaxis] + expected_requests_1)

    def _make_transition_matrix(self, location_idx):
        """
        Creates likelihood matrix P(Ni' | Ni''), where:
        - Ni'' = number of cars available at start of day at location i
        - Ni' = number of cars available at end of day at location i
        for all possible values of Ni'' and Ni', in the inclusive interval [0, max_cars].

        P(Ni' | Ni'') = sum_over_possible_Xs(p(X_i|lam_i, N_i'') * p(Y_i = Ni' - Ni'' + X_i|mu_i, N_max - Ni'' + X_i))
        where:
        - X_i = number of rentals at location i
        - Y_i = number of returns at location i
        - lam_i = mean number of rentals at location i
        - mu_i = mean number of returns at location i
        """
        max_cars = self.max_cars
        P = np.zeros((max_cars + 1, max_cars + 1))
        N = np.arange(max_cars + 1)
        px = []
        py = []
        for n in N:
            px.append(get_truncated_poisson_pmf(high=n, lam=self.locations[location_idx].mean_requests))
            py.append(get_truncated_poisson_pmf(high=n, lam=self.locations[location_idx].mean_returns))

        for n_start in N:
            px_n_start = px[n_start]
            for n_end in N:
                for x in range(n_start + 1):
                    py_n_end = py[max_cars - n_start + x]
                    y = n_end - n_start + x
                    if y < 0:
                        # Don't add to the likelihood matrix if the number of returns is less than 0
                        continue
                    P[n_end, n_start] += px_n_start[x] * py_n_end[y]

        # Check sum_over_Ni'(P(Ni' | Ni'')) = 1
        assert np.allclose(np.sum(P, axis=0), 1)

        return P

    def _init_transition_probs(self):
        """
        Creates likelihood matrices P(Ni' | Ni''), where:
        - Ni'' = number of cars available at start of day at location i
        - Ni' = number of cars available at end of day at location i
        for all possible values of Ni'' and Ni', in the inclusive interval [0, max_cars].

        P(Ni' | Ni'') = sum_over_possible_Xs(p(X_i|lam_i, N_i'') * p(Y_i = Ni' - Ni'' + X_i|mu_i, N_max - Ni'' + X_i))
        where:
        - X_i = number of rentals at location i
        - Y_i = number of returns at location i
        - lam_i = mean number of rentals at location i
        - mu_i = mean number of returns at location i
        """
        self.prob_matrices = {}
        for location_idx in range(len(self.locations)):
            self.prob_matrices[location_idx] = self._make_transition_matrix(location_idx)

    def get_expected_value(self, value, gamma):
        trans_prob_1 = self.prob_matrices[0]
        trans_prob_2 = self.prob_matrices[1]
        expected_reward = self.expected_rental_reward
        expected_next_value = gamma * (trans_prob_1.T.dot(value).dot(trans_prob_2))
        expected_value_matrix = expected_reward + expected_next_value
        return expected_value_matrix

    def compute_next_state(self, state, action):
        state_1, state_2 = state
        state_1_morning = state_1 - action
        state_2_morning = state_2 + action

        if 0 <= state_1_morning <= self.max_cars and 0 <= state_2_morning <= self.max_cars:
            return (state_1_morning, state_2_morning)
        else:
            return None


def main():

    ## Testing Jack's Car Rental environment
    # Create the environment
    env = JacksCarRental()
    # Get the expected reward for a given state and action
    state = (10, 10)
    action = 3
    expected_reward = env.get_expected_reward_(state, action)
    print(f"Expected reward for state {state} and action {action} is {expected_reward}")
    # Get the probability of transitioning from state to next_state, given action
    next_state = (5, 15)
    prob = env.get_state_transition_probs_(next_state, state, action)
    print(f"Probability of transitioning from state {state} to state {next_state} given action {action} is {prob}")


if __name__ == "__main__":
    main()
