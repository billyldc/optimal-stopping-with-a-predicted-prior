import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt


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


def lambdas(beta):
    """Compute lambda_1 and lambda_2 for a given beta."""
    if beta < 0 or beta > 1 / np.e:
        raise ValueError("Beta must be in the range [0, 1/e]")
    if beta == 0:
        return 0, 1
    lambda1 = root_scalar(
        lambda x: -x * np.log(x) - beta, bracket=(1e-10, 1 / np.e), method="brentq"
    ).root
    lambda2 = root_scalar(
        lambda x: -x * np.log(x) - beta, bracket=(1 / np.e, 1 - 1e-10), method="brentq"
    ).root
    return lambda1, lambda2


def plot_thresholds(ax, T, D, GM, lambda_1, lambda_2):
    """Plot Dynkin, Gilbert-Mosteller, and Robust thresholds on the given axis."""
    R = np.where((T >= lambda_1) & (T <= lambda_2), GM, D)
    ax.plot(T, D, label="Dynkin", color="tab:orange")
    ax.plot(T, GM, label="Gilbert-Mosteller", color="tab:blue")
    ax.plot(T, R, label="Robust", color="tab:green", linewidth=3)


def style_threshold_plot(ax, lambda_1, lambda_2):
    """Apply consistent styling to the threshold plot."""
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\theta(t)$", labelpad=-5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xticks = [lambda_1, lambda_2, 1]
    xtick_labels = [f"${val:.3f}$" if i < 2 else f"${val:.0f}$" for i, val in enumerate(xticks)]
    yticks = [1]
    ytick_labels = [f"${val:.0f}$" for val in yticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    ax.text(-0.02, -0.02, r"$0$", ha="right", va="top")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc=(0.8, 0.8))


def main():
    n = 10
    beta = 1 / 3
    T = np.linspace(0, 1 - 1e-9, 10000)
    D = dynkin_threshold(T)
    GM = gilbert_mosteller_threshold(T, n)

    lambda_1, lambda_2 = lambdas(beta)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    plot_thresholds(ax, T, D, GM, lambda_1, lambda_2)
    style_threshold_plot(ax, lambda_1, lambda_2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
