"""
Jack's Car Rental environment - from Sutton and Barto, 2018, p. 81.

TODO:
  - Random seed, for reproducibility
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


class RentalLocation:
    def __init__(
            self,
            expected_rentals,
            expected_returns,
            max_cars=20,
            max_move_cars=5,
            rental_reward=10,
            move_cost=2,
            initial_cars=None,
            name=None,
    ):

        self.max_cars = max_cars
        self.max_move_cars = max_move_cars
        self.expected_rentals = expected_rentals
        self.expected_returns = expected_returns
        self.rental_reward = rental_reward
        self.move_cost = move_cost
        self.name = name

        if initial_cars is None:
            self.cars = np.random.randint(0, max_cars + 1)
        else:
            self.cars = initial_cars

    def get_car_returns(self):
        """
        - Determines the number of cars returned to this location - an int from a Poisson distribution with mean
        expected_returns.
        - Updates the number of cars at this location.

        Returns:
            int: The number of cars returned.
        """
        returned_cars = np.random.poisson(self.expected_returns)
        # Max returnable cars is capped to the difference between the max number of cars and the current number
        returned_cars = min(returned_cars, self.max_cars - self.cars)

        # Update number of cars
        self.cars += returned_cars

    def get_rentals(self):
        """
        Get the number of cars rented from this location.

        Returns:
            int: The number of cars rented.
        """
        rentals = np.random.poisson(self.expected_rentals)
        rentals = min(rentals, self.cars)

        # Update number of cars
        self.cars -= rentals

        return rentals

    def move_cars(self, action):
        """
        Move cars from this location to another.

        Args:
            action (int): The number of cars to move; positive => FROM this location TO another location.
        """
        # The number of cars to move is capped by the number of cars at this location and the max cars moved
        # - e.g. 1: want to move 4 cars, but there are only 3 cars at this location => move 3 cars
        # - e.g. 2: want to move 6 cars, but the max cars moved is 5 => move 5 cars
        num_cars = min(action, self.cars, self.max_move_cars)
        self.cars -= num_cars
        return num_cars

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action (int): The number of cars to move; positive => FROM this location TO another location.

        Returns:
            int: The reward for the step.
        """
        # Get returned cars from the previous day (poisson distributed) and update the car count
        self.get_car_returns()

        # TODO: reorder so move car decision before random return? Harder problem?
        # Move cars overnight, before business hours
        num_cars_moved = self.move_cars(action)

        # Get rentals during the day
        num_rentals = self.get_rentals()

        # Calculate reward
        reward = self.rental_reward * num_rentals - self.move_cost * num_cars_moved

        # Package secondary metrics for logging
        info = {"num_rentals": num_rentals, "num_cars_moved": num_cars_moved}

        return reward, info


class JacksCarRental:

    def __init__(
            self,
            location_1_params=None,
            location_2_params=None,
    ):

        # Default parameters
        if location_2_params is None:
            location_2_params = {"expected_rentals": 4, "expected_returns": 2, "initial_cars": 10}
        if location_1_params is None:
            location_1_params = {"expected_rentals": 3, "expected_returns": 3, "initial_cars": 10}

        self.rental_location_1 = RentalLocation(
            expected_rentals=location_1_params["expected_rentals"],
            expected_returns=location_1_params["expected_returns"],
            initial_cars=location_1_params["initial_cars"],
            name="location_1",
        )
        self.rental_location_2 = RentalLocation(
            expected_rentals=location_2_params["expected_rentals"],
            expected_returns=location_2_params["expected_returns"],
            initial_cars=location_2_params["initial_cars"],
            name="location_2",
        )

    def step(self, action):
        """
        Take a step in the environment. Runs the step functions for each rental location, gets the reward (rentals
        less moving costs) for each location, then returns the combined reward.

        Args:
            action (int): The number of cars to move from location 1 to location 2 (+ve if net efflux, -ve otherwise).
        """

        reward = 0
        num_rentals = {"location_1": 0, "location_2": 0}
        num_cars_moved = {"location_1": 0, "location_2": 0}

        # Step through each location
        for rental_location in [self.rental_location_1, self.rental_location_2]:

            name = rental_location.name

            # Step through each location
            step_reward, info = rental_location.step(action)
            reward += step_reward

            # Log secondary metrics
            num_rentals[name] = info["num_rentals"]
            num_cars_moved[name] = info["num_cars_moved"]
            info = {"num_rentals": num_rentals, "num_cars_moved": num_cars_moved}

        return reward, info


def main():

    """
    Test function, which checks the Jack's Car Rental environment runs correctly.

    - Creates an environment with default parameters.
    - Runs 10 steps, with random actions.
    - Plots the number of cars at each location after each step.
    - Plots the reward for each step.
    """

    # Create environment
    env = JacksCarRental(
        location_1_params={"expected_rentals": 3, "expected_returns": 3, "initial_cars": 10},
        location_2_params={"expected_rentals": 4, "expected_returns": 2, "initial_cars": 10},
    )

    # Run 10 steps
    num_steps = 10
    rewards = []
    cars_1 = []
    cars_2 = []
    rented_cars_1 = []
    rented_cars_2 = []
    moved_cars_1 = []
    moved_cars_2 = []
    for _ in range(num_steps):
        action = np.random.randint(-5, 6)
        reward, info = env.step(action)
        rewards.append(reward)
        cars_1.append(env.rental_location_1.cars)
        cars_2.append(env.rental_location_2.cars)
        rented_cars_1.append(info["num_rentals"]["location_1"])
        rented_cars_2.append(info["num_rentals"]["location_2"])
        moved_cars_1.append(info["num_cars_moved"]["location_1"])
        moved_cars_2.append(info["num_cars_moved"]["location_2"])
    total_rented = np.array(rented_cars_1) + np.array(rented_cars_2)

    # Plot results
    # A single figure with 2 subplots: subplot 1: cars at location 1 and 2; subplot 2: reward
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(cars_1, label="Location 1")
    axs[0].plot(cars_2, label="Location 2")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Number of cars")
    axs[0].legend()
    axs[1].plot(rewards, label="Reward")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Reward")
    axs[2].plot(rented_cars_1, label="Rented cars from location 1")
    axs[2].plot(rented_cars_2, label="Rented cars from location 2")
    axs[2].plot(total_rented, label="Total rented cars")
    axs[2].plot(moved_cars_1, label="Moved cars from location 1")
    axs[2].plot(moved_cars_2, label="Moved cars from location 2")
    axs[2].set_xlabel("Step")
    axs[2].set_ylabel("Number of cars rented/moved")
    axs[2].legend()
    plt.show()


if __name__ == "__main__":
    main()
