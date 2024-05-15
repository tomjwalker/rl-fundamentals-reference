import numpy as np


A = np.array([
    [1, -0.5, 0, 0, 0],
    [-0.5, 1, -0.5, 0, 0],
    [0, -0.5, 1, -0.5, 0],
    [0, 0, -0.5, 1, -0.5],
    [0, 0, 0, -0.5, 1],
])

b = np.array([0, 0, 0, 0, -0.5])

x = np.linalg.solve(A, -b)

print(x)
