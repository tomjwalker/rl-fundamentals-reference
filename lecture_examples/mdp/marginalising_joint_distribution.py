import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import matplotlib
matplotlib.use('TkAgg')


# Define the mean and covariance matrix
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]

# Create a grid of (x, y) points
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Create the bivariate normal distribution
rv = multivariate_normal(mean, cov)
Z = rv.pdf(pos)

# Marginal distributions
marginal_x = np.sum(Z, axis=0) * (y[1] - y[0])
marginal_y = np.sum(Z, axis=1) * (x[1] - x[0])

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the bivariate normal distribution
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Plot the marginal distributions
ax.plot(x, np.full_like(x, 3), marginal_x, color='r', lw=2, label='Marginal X')
ax.plot(np.full_like(y, -3), y, marginal_y, color='b', lw=2, label='Marginal Y')

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Probability Density')
ax.set_title('Bivariate Normal Distribution with Marginal Distributions')
ax.legend()

# Show plot
plt.show()
