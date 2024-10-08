import argparse
import matplotlib
import matplotlib.pyplot as plt
import random
import math
import sys
from typing import List, Tuple

matplotlib.use('TkAgg')


def exact_pentagram_area(radius_of_enclosing_circle: float = 1) -> float:
    """
    Calculates the exact area of a regular pentagram inscribed in a circle.

    Args:
        radius_of_enclosing_circle (float): The radius of the circle in which the pentagram is inscribed.

    Returns:
        float: The exact area of the pentagram.
    """
    d = radius_of_enclosing_circle / (
        math.sin(math.pi / 5) * math.tan(2 * math.pi / 5) + math.cos(math.pi / 5)
    )

    area_pentagon = (
        5 * (d ** 2) * math.sin(math.pi / 5) * math.cos(math.pi / 5)
    )

    area_stellation = (
        (d ** 2) * (math.sin(math.pi / 5) ** 2) * math.tan(2 * math.pi / 5)
    )

    return area_pentagon + 5 * area_stellation


def complex_cross_product(z1: complex, z2: complex) -> float:
    """
    Calculates an analog to the cross product for complex numbers in 2D.

    Args:
        z1 (complex): A complex number representing a 2D vector.
        z2 (complex): Another complex number representing a 2D vector.

    Returns:
        float: A real number representing the signed area of the parallelogram
            formed by z1 and z2. Positive indicates z2 is to the left of z1,
            negative implies z2 is to the right.
    """
    return (z1.real * z2.imag) - (z1.imag * z2.real)


def is_inside_pentagram(x: float, y: float) -> bool:
    """
    Checks if a point (x, y) is inside a regular pentagram inscribed in a unit circle.

    Args:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.

    Returns:
        bool: True if the point is inside the pentagram, False otherwise.
    """
    # Calculate vertices of the pentagram
    vertices: List[complex] = [
        math.cos(2 * math.pi * i / 5) + 1j * math.sin(2 * math.pi * i / 5)
        for i in range(5)
    ]

    # Check if the point is inside the pentagram
    cross_products: List[float] = []
    for i in range(5):
        v1: complex = vertices[i]
        v2: complex = vertices[(i + 2) % 5]  # Next vertex for pentagram edges

        segment: complex = v2 - v1
        point: complex = (x + 1j * y) - v1
        cross_product: float = complex_cross_product(segment, point)
        cross_products.append(cross_product)

    # Case 1: Inside the inner pentagon
    if all(cp > 0 for cp in cross_products):
        return True

    # Case 2: Inside the stellations
    for i in range(5):
        if cross_products[i] < 0:
            if cross_products[(i + 1) % 5] > 0 and cross_products[(i - 1) % 5] > 0:
                return True

    return False


def monte_carlo_pentagram_area(
    num_samples: int = 10000
) -> Tuple[float, List[float], List[Tuple[Tuple[float, float], bool]]]:
    """
    Estimates the area of a pentagram using the Monte Carlo method.

    Args:
        num_samples (int): The number of random samples to use in the simulation.

    Returns:
        Tuple[float, List[float], List[Tuple[Tuple[float, float], bool]]]:
            - The final estimated area.
            - A list of area estimates over iterations.
            - A log of sampled points and their inclusion status.
    """
    inside_count: int = 0
    log: List[Tuple[Tuple[float, float], bool]] = []
    area_estimates: List[float] = []
    # HOMEWORK: input the area of the square bounding the unit circle (i.e. whose radius is 1)
    area_square: float = 4.0  # Area of the square bounding the unit circle

    # Sample points within the square bounding the unit circle
    for num in range(num_samples):
        x: float = random.uniform(-1, 1)
        y: float = random.uniform(-1, 1)

        # HOMEWORK: use the helper function to check if the point is inside the pentagram
        inside: bool = is_inside_pentagram(x, y)

        # HOMEWORK STARTS: if the point is inside the pentagram, increment the inside_count (1-2 lines)
        if inside:
            inside_count += 1
        # HOMEWORK ENDS

        log.append(((x, y), inside))

        # HOMEWORK: Area estimate is the ratio of points inside the pentagram to the total points sampled
        # N.B. `num` is 0-indexed, so we add 1 to the denominator to avoid division by zero
        # N.B., needs to be multiplied by the area of the square bounding the unit circle
        area_estimate: float = (inside_count / (num + 1)) * area_square

        # HOMEWORK: append current area estimate to list of area estimates (this will provide a history of estimates)
        area_estimates.append(area_estimate)

    final_estimate: float = area_estimates[-1]
    return final_estimate, area_estimates, log


def plot_results(
    num_samples: int,
    area_estimates: List[float],
    exact_area: float,
    log: List[Tuple[Tuple[float, float], bool]],
):
    """
    Plots the convergence of the estimated area and the scatter plot of sampled points.

    Args:
        num_samples (int): The number of samples used in the simulation.
        area_estimates (List[float]): The list of area estimates over iterations.
        exact_area (float): The exact area of the pentagram.
        log (List[Tuple[Tuple[float, float], bool]]): The log containing sampled points and their status.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot convergence of area estimates
    ax[0].plot(range(num_samples), area_estimates, color="blue", label="Estimated Area")
    ax[0].axhline(y=exact_area, color="black", linestyle="--", label="Exact Area")
    ax[0].set_xlabel("Number of Samples")
    ax[0].set_ylabel("Estimated Area")
    ax[0].set_title(
        f"MC Estimation (exact area: {round(exact_area, 3)}, final est: {round(area_estimates[-1], 3)})"
    )
    ax[0].legend()

    # Plot scatter of sampled points
    colors = ["#22a884" if inside else "#c6c6c6" for (_, inside) in log]
    x_coords = [point[0] for (point, _) in log]
    y_coords = [point[1] for (point, _) in log]
    ax[1].scatter(x_coords, y_coords, color=colors, alpha=0.5, s=4)
    ax[1].set_aspect('equal', adjustable='datalim')
    ax[1].set_title("Monte Carlo Sampling of the Pentagram")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run the Monte Carlo simulation and plot the results.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Estimate the area of a pentagram using the Monte Carlo method."
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10000,
        help='Number of random samples to use in the simulation (default: 10000)'
    )
    args = parser.parse_args()

    # Validate num_samples
    if args.num_samples <= 0:
        print("Error: Number of samples must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    num_samples: int = args.num_samples

    # Calculate the exact area
    exact_area: float = exact_pentagram_area()

    # Run the Monte Carlo simulation
    final_estimate, area_estimates, log = monte_carlo_pentagram_area(num_samples)

    print(f"Final estimated area of pentagram: {final_estimate:.6f}")
    print(f"Exact area of pentagram: {exact_area:.6f}")

    # Plot the results
    plot_results(num_samples, area_estimates, exact_area, log)


if __name__ == "__main__":
    main()
