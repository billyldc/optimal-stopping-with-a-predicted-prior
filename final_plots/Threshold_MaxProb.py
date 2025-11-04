import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from helper import compute_λ, setup_threshold_plot


def dynkin_threshold(T):
    """Compute Dynkin's threshold function."""
    return [1 if t < 1 / np.e else 0 for t in T]


def gilbert_mosteller_threshold(T, n):
    """
    Compute Gilbert-Mosteller thresholds for a range of t values given n.
    For each t, we need to find the root of the equation:
    ∫[t, 1] ( (1-s+sx)^(n-1) - x^(n-1) ) / (1-s) ds - x^(n-1) = 0
    """

    def integrand(t, x):
        return ((1 - t + t * x) ** (n - 1) - x ** (n - 1)) / (1 - t)

    def equation(x, t, n):
        I, _ = quad(integrand, t, 1, args=(x,))
        return I - x ** (n - 1)

    thresholds = []
    for t in T:
        init = 0.5
        sol = root_scalar(
            equation, args=(t, n), bracket=[0, 1], method="brentq", x0=init
        )
        thresholds.append(sol.root)
    return thresholds


def plot_thresholds(n, β):
    """Plot thresholds for given β and n."""
    T = np.linspace(0, 1 - 1e-9, 10000)
    D = dynkin_threshold(T)
    GM = gilbert_mosteller_threshold(T, n)

    λ1, λ2 = compute_λ(β)
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=500)

    R = np.where((T >= λ1) & (T <= λ2), GM, D)
    ax.plot(T, D, label="Dynkin", color="tab:orange")
    ax.plot(T, GM, label="Gilbert-Mosteller", color="tab:blue")
    ax.plot(T, R, label="Robust", color="tab:green", linewidth=3)

    setup_threshold_plot(ax, λ1, λ2)


if __name__ == "__main__":
    n = 10
    β = 1 / 3
    plot_thresholds(n, β)
    plt.tight_layout()
    plt.show()
