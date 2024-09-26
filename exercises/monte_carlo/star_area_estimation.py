import matplotlib
import matplotlib.pyplot as plt

import random
import math
import numpy as np
matplotlib.use('TkAgg')


def exact_pentagram_area(radius_of_enclosing_circle=1):

    d_pentagon_vertex_to_centre = radius_of_enclosing_circle / (math.sin(math.pi / 5) * math.tan(2 * math.pi / 5) +
                                                                math.cos(math.pi / 5))

    area_pentagon = 5 * (d_pentagon_vertex_to_centre ** 2) * math.sin(math.pi / 5) * math.cos(math.pi / 5)

    area_stellation = (d_pentagon_vertex_to_centre ** 2) * (math.sin(math.pi / 5) ** 2) * math.tan(2 * math.pi / 5)

    return area_pentagon + 5 * area_stellation


def complex_cross_product(z1, z2):
    """Calculates an analog to the cross product for complex numbers in 2D.

    Args:
        z1: A complex number representing a 2D vector.
        z2: Another complex number representing a 2D vector.

    Returns:
        A real number representing the signed area of the parallelogram 
        formed by z1 and z2. Positive indicates z2 is to the left of z1, 
        negative implies z2 is to the right.
    """
    return (z1.real * z2.imag) - (z1.imag * z2.real)


def is_inside_pentagram(x, y):
    """Checks if a point (x, y) is inside a regular pentagram inscribed in a unit circle."""

    # Calculate vertices of the pentagram (complex number math is helpful here)
    vertices = [math.cos(2 * math.pi * i / 5) + 1j * math.sin(2 * math.pi * i / 5)
                for i in range(5)]

    # Case 1: is inside pentagram if it is inside the inner pentagon. This is true if point is to the left of all
    # segments of the pentagram
    cross_products = []
    for i in range(5):
        v1 = vertices[i]
        # For a pentagram, the next vertex is 2 vertices away
        v2 = vertices[(i + 2) % 5]    # Wrap around to the start if needed

        # Use cross product to determine if the point is on the correct side of the edge (if cross product is positive,
        # point is on the left)
        segment = v2 - v1
        point = (x + 1j * y) - v1
        cross_product = complex_cross_product(segment, point)
        cross_products.append(cross_product)

    if all(cross_product > 0 for cross_product in cross_products):
        return True

    # Case 2: For the stellations, the point can still be inside if it is to the right of a given segment but to the
    # left of neighbouring segments on either side.

    # Already have the cross products for the pentagram, so just need to check the stellations
    for i in range(5):
        if cross_products[i] < 0:
            # Check if point is to the left of both neighbouring segments
            if cross_products[(i + 1) % 5] > 0 and cross_products[(i - 1) % 5] > 0:
                return True

    return False


# Monte Carlo Simulation
num_samples = 10000  # Adjust for desired accuracy
inside_count = 0
log = []
exact_area = exact_pentagram_area(radius_of_enclosing_circle=1)
area_estimates = []
area_square = 2 ** 2
for num_samples_intermediate in range(num_samples):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)

    if is_inside_pentagram(x, y):
        inside_count += 1

    log.append([(x, y), is_inside_pentagram(x, y)])

    area_estimate = (inside_count / (num_samples_intermediate + 1)) * area_square
    area_estimates.append(area_estimate)

print("Final estimated area of pentagram:", area_estimate)

# Plotting

fig, ax = plt.subplots(1, 2)

ax[0].plot(range(num_samples), area_estimates, color="blue")

# Add a horizontal line for the exact area
ax[0].axhline(y=exact_area, color="black", linestyle="--")

# Display final estimate (to 3sf) alongside the exact value in the axis title
ax[0].set_title(f"Estimated Area: {area_estimate:.3f} (Exact: {exact_area:.3f})")

colors = ["#22a884" if inside else "#c6c6c6" for (point, inside) in log]
ax[1].scatter([point[0] for (point, inside) in log], [point[1] for (point, inside) in log], color=colors, alpha=0.5)

# Make axis 1 axes equal
ax[1].set_aspect('equal', adjustable='datalim')

plt.show()
