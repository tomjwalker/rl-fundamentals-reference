import numpy as np
from scipy.stats import poisson
import matplotlib
matplotlib.use('TkAgg')

from typing import List, Tuple, Optional, Dict


def get_truncated_poisson_pmf(high: int, lam: float) -> np.ndarray:
    """
    Compute the probability mass function for a truncated Poisson distribution from 0 to 'high', inclusive.

    The probability of the event X = high is adjusted so that the total probability sums to 1.

    Args:
        high (int): The upper bound of the truncated Poisson distribution.
        lam (float): The lambda (mean rate) of the Poisson distribution.

    Returns:
        np.ndarray: An array of probabilities for each value from 0 to 'high' inclusive.
    """
    # Get the pmf for a Poisson distribution with lambda=lam
    pmf = poisson.pmf(np.arange(high + 1), lam)
    # Get the sum of all probabilities up to, but excluding, 'high'
    prob = np.sum(pmf[:high])
    # Adjust the probability at 'high' to ensure the total sums to 1
    pmf[high] = 1 - prob
    return pmf


class Location:
    def __init__(self, mean_requests: float, mean_returns: float) -> None:
        """
        Initialise a rental location.

        Args:
            mean_requests (float): The mean number of rental requests at this location per day.
            mean_returns (float): The mean number of returns at this location per day.
        """
        self.mean_requests: float = mean_requests
        self.mean_returns: float = mean_returns


class JacksCarRental:
    def __init__(
        self,
        max_cars: int = 20,
        max_move_cars: int = 5,
        rental_reward: float = 10,
        move_cost: float = 2,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Initialise the Jack's Car Rental environment.

        Args:
            max_cars (int): The maximum number of cars at each location.
            max_move_cars (int): The maximum number of cars that can be moved overnight.
            rental_reward (float): The reward per car rented.
            move_cost (float): The cost per car moved overnight.
            random_seed (Optional[int]): The seed for random number generation.
        """
        self.max_cars: int = max_cars
        self.max_move_cars: int = max_move_cars
        self.rental_reward: float = rental_reward
        self.move_cost: float = move_cost
        self.random_seed: Optional[int] = random_seed

        self.locations: List[Location] = []
        self.expected_rental_reward: np.ndarray = np.zeros((self.max_cars + 1, self.max_cars + 1))
        self.prob_matrices: Dict[int, np.ndarray] = {}
        self._init_locations()
        self._init_expected_rental_reward()
        self._init_transition_probs()

    def _init_locations(self) -> None:
        """
        Initialise the rental locations with their mean rental requests and returns.
        """
        self.locations = [
            Location(mean_requests=3, mean_returns=3),
            Location(mean_requests=4, mean_returns=2),
        ]

    def _get_expected_reward_for_location(self, location_idx: int) -> np.ndarray:
        """
        Calculate the expected number of rentals at a location for all possible starting car counts.

        Args:
            location_idx (int): The index of the location (0 or 1).

        Returns:
            np.ndarray: An array of expected rentals for each possible starting number of cars.
        """
        possible_rentals = np.arange(self.max_cars + 1)
        expected_requests = np.zeros(len(possible_rentals))

        for possible_rentals_max in possible_rentals:
            # Probability of X rentals given up to 'possible_rentals_max' cars are available
            prob_to_max = get_truncated_poisson_pmf(
                high=possible_rentals_max,
                lam=self.locations[location_idx].mean_requests
            )
            # Expected rentals given the available cars
            expected_requests[possible_rentals_max] = np.sum(
                possible_rentals[:possible_rentals_max + 1] * prob_to_max
            )

        return expected_requests

    def _init_expected_rental_reward(self) -> None:
        """
        Calculate and store the expected rental rewards for all possible states.

        The expected rental reward is calculated for all combinations of cars at location 1 and location 2.
        """
        expected_requests_0 = self._get_expected_reward_for_location(0)
        expected_requests_1 = self._get_expected_reward_for_location(1)

        self.expected_rental_reward = self.rental_reward * (
            expected_requests_0[:, np.newaxis] + expected_requests_1
        )

    def _make_transition_matrix(self, location_idx: int) -> np.ndarray:
        """
        Create the state transition probability matrix P(Ni' | Ni'') for a location.

        Args:
            location_idx (int): The index of the location (0 or 1).

        Returns:
            np.ndarray: A (max_cars + 1) x (max_cars + 1) matrix of transition probabilities.
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

        # Ensure that the probabilities sum to 1 over Ni'
        assert np.allclose(np.sum(P, axis=0), 1), "Transition probabilities do not sum to 1."

        return P

    def _init_transition_probs(self) -> None:
        """
        Initialise the state transition probability matrices for both locations.
        """
        for location_idx in range(len(self.locations)):
            self.prob_matrices[location_idx] = self._make_transition_matrix(location_idx)

    def get_expected_value(self, value: np.ndarray, gamma: float) -> np.ndarray:
        """
        Calculate the expected value for all states.

        Args:
            value (np.ndarray): The current value function, a (max_cars + 1) x (max_cars + 1) array.
            gamma (float): The discount factor.

        Returns:
            np.ndarray: An array of expected values for all states.
        """
        trans_prob_1 = self.prob_matrices[0]
        trans_prob_2 = self.prob_matrices[1]
        expected_reward = self.expected_rental_reward
        expected_next_value = gamma * (trans_prob_1.T @ value @ trans_prob_2)
        expected_value_matrix = expected_reward + expected_next_value
        return expected_value_matrix

    def compute_next_state(self, state: Tuple[int, int], action: int) -> Optional[Tuple[int, int]]:
        """
        Compute the next state after moving cars overnight.

        Args:
            state (Tuple[int, int]): The current state (number of cars at each location).
            action (int): The number of cars moved from location 1 to location 2 overnight.
                          Positive values move cars from location 1 to 2, negative values move cars from 2 to 1.

        Returns:
            Optional[Tuple[int, int]]: The next state if the action is valid, otherwise None.
        """
        state_1, state_2 = state
        state_1_morning = state_1 - action
        state_2_morning = state_2 + action

        if 0 <= state_1_morning <= self.max_cars and 0 <= state_2_morning <= self.max_cars:
            return int(state_1_morning), int(state_2_morning)
        else:
            return None


def main() -> None:
    # Testing Jack's Car Rental environment
    # Create the environment
    env = JacksCarRental()

    # Get the expected value for a given value function
    # Let's create a dummy value function, e.g., zeros
    value = np.zeros((env.max_cars + 1, env.max_cars + 1))

    # Get the expected value matrix
    gamma = 0.9
    expected_value_matrix = env.get_expected_value(value, gamma)
    print(f"Expected value matrix shape: {expected_value_matrix.shape}")
    print(f"Expected value at state (10, 10): {expected_value_matrix[10, 10]}")

    # Compute the next state for a given state and action
    state = (10, 10)
    action = 3
    next_state = env.compute_next_state(state, action)
    print(f"Next state from state {state} with action {action}: {next_state}")

    # Check if the next state is valid
    if next_state is not None:
        print(f"Valid next state: {next_state}")
    else:
        print(f"Action {action} from state {state} is invalid.")

    # Access the expected rental reward matrix
    expected_rental_reward = env.expected_rental_reward
    print(f"Expected rental reward at state (10, 10): {expected_rental_reward[10, 10]}")

    # Access the state transition probability matrices
    trans_prob_1 = env.prob_matrices[0]
    trans_prob_2 = env.prob_matrices[1]
    # Transition probability from n_start to n_end at location 1
    n_start = 10
    n_end = 5
    prob_loc1 = trans_prob_1[n_end, n_start]
    print(f"Probability of moving from {n_start} to {n_end} cars at location 1: {prob_loc1}")
    # Similarly for location 2
    prob_loc2 = trans_prob_2[n_end, n_start]
    print(f"Probability of moving from {n_start} to {n_end} cars at location 2: {prob_loc2}")


if __name__ == "__main__":
    main()
