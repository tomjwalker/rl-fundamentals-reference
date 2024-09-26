import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def plot_weight_decay(alpha, n_max):
    n = np.arange(1, n_max + 1)

    # Plot (1 - alpha)^n
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(n, (1 - alpha) ** n)
    plt.title(f'Weight of initial estimate: (1 - α)^n, α = {alpha}')
    plt.xlabel('n')
    plt.ylabel('Weight')

    # Plot α(1 - α)^(n - i) for different i
    plt.subplot(122)
    for i in [1, 5, 10, 20]:
        weights = alpha * (1 - alpha) ** (n - i)
        plt.plot(n[i - 1:], weights[i - 1:], label=f'i = {i}')

    plt.title(f'Weight of rewards: α(1 - α)^(n - i), α = {alpha}')
    plt.xlabel('n')
    plt.ylabel('Weight')
    plt.legend()
    plt.tight_layout()
    plt.show()


# Example usage
plot_weight_decay(alpha=0.5, n_max=50)