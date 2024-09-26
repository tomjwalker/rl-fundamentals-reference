import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

import numpy as np

GAMMA = 0.5

# =============================
# Deterministic // equiprobable
# =============================


A = np.array([
    [(1 - GAMMA/2), (-GAMMA/4), (-GAMMA/4)],
    [(-GAMMA/4), (1 - GAMMA/2), 0],
    [(-GAMMA/4), 0, (1 - GAMMA/2)],
])

b = np.array([0, 1/4, 1/4])

determinant = (1 - GAMMA/2) * (1 - GAMMA + GAMMA**2/8)

A_adj = np.array([
    [(1 - GAMMA/2)**2, (GAMMA/4 * (1 - GAMMA/2)), (GAMMA/4 * (1 - GAMMA/2))],
    [(GAMMA/4 * (1 - GAMMA/2)), ((1 - GAMMA/2)**2 - (GAMMA/4)**2), (GAMMA/4)**2],
    [(GAMMA/4 * (1 - GAMMA/2)), (GAMMA/4)**2, ((1 - GAMMA/2)**2 - (GAMMA/4)**2)],
])

A_inv = A_adj / determinant

v = np.dot(A_inv, b)

print(f"Deterministic env; equiprobable policy {v}")

# ==========================
# Stochastic // equiprobable
# ==========================

A = np.array([
    [(1 - GAMMA/2), (-GAMMA/4), (-GAMMA/4)],
    [(-GAMMA/4), (1 - GAMMA/2), 0],
    [(-GAMMA/4), 0, (1 - GAMMA/2)],
])

b = np.array([0, 1/4, 1/4])

determinant = (1 - GAMMA/2) * (1 - GAMMA + GAMMA**2/8)

A_adj = np.array([
    [(1 - GAMMA/2)**2, (GAMMA/4 * (1 - GAMMA/2)), (GAMMA/4 * (1 - GAMMA/2))],
    [(GAMMA/4 * (1 - GAMMA/2)), ((1 - GAMMA/2)**2 - (GAMMA/4)**2), (GAMMA/4)**2],
    [(GAMMA/4 * (1 - GAMMA/2)), (GAMMA/4)**2, ((1 - GAMMA/2)**2 - (GAMMA/4)**2)],
])

A_inv = A_adj / determinant

v = np.dot(A_inv, b)

print(f"Stochastic env; equiprobable policy {v}")


# ==========================
# Stochastic // final policy
# ==========================


# TODO
# A = np.array([
#     [(1 - GAMMA/3), (-GAMMA/3), (-GAMMA/3)],
#     [0, (1 - 2 * GAMMA/3), 0],
#     [0, 0, (1 - 2 * GAMMA/3)],
# ])
#
# b = np.array([0, 1/3, 1/3])
#
# v = np.linalg.solve(A, b)
#
# print(f"Stochastic env; final policy {v}")
