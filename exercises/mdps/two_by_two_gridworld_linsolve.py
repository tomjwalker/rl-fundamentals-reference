# TODO: clean up
# TODO: change slightly


import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

import numpy as np


# =============================
# Deterministic // equiprobable
# =============================

GAMMA = 0.5

A = np.array([
    [(1 - GAMMA/2), (-GAMMA/4), (-GAMMA/4)],
    [(-GAMMA/4), (1 - GAMMA/2), 0],
    [(-GAMMA/4), 0, (1 - GAMMA/2)],
])

b = np.array([0, 1/4, 1/4])

v = np.linalg.solve(A, b)

print(f"Deterministic env; equiprobable policy {v}")


# =============================
# Deterministic // final policy
# =============================

A = np.array([
    [1, -GAMMA/2, -GAMMA/2],
    [0, 1, 0],
    [0, 0, 1],
])

b = np.array([0, 1, 1])

v = np.linalg.solve(A, b)

print(f"Deterministic env; final policy {v}")


# ==========================
# Stochastic // equiprobable
# ==========================

A = np.array([
    [(1 - GAMMA/2), (-GAMMA/4), (-GAMMA/4)],
    [(-GAMMA/4), (1 - GAMMA/2), 0],
    [(-GAMMA/4), 0, (1 - GAMMA/2)],
])

b = np.array([0, 1/4, 1/4])

v = np.linalg.solve(A, b)

print(f"Stochastic env; equiprobable policy {v}")


# ==========================
# Stochastic // final policy
# ==========================

A = np.array([
    [(1 - GAMMA/3), (-GAMMA/3), (-GAMMA/3)],
    [0, (1 - 2 * GAMMA/3), 0],
    [0, 0, (1 - 2 * GAMMA/3)],
])

b = np.array([0, 1/3, 1/3])

v = np.linalg.solve(A, b)

print(f"Stochastic env; final policy {v}")
