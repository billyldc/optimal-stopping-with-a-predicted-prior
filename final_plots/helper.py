import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.integrate import quad


def compute_λ(β):
    """Compute lambda_1 and lambda_2 for a given β."""
    if β < 0 or β > 1 / np.e:
        raise ValueError("Beta must be in the range [0, 1/e]")
    if β == 0:
        return 0, 1
    if β == 1 / np.e:
        return 1 / np.e, 1 / np.e

    root1 = root_scalar(
        lambda x: -x * np.log(x) - β, bracket=(1e-10, 1 / np.e), method="brentq"
    )
    λ1 = root1.root if root1.converged else None

    root2 = root_scalar(
        lambda x: -x * np.log(x) - β, bracket=(1 / np.e, 1 - 1e-10), method="brentq"
    )
    λ2 = root2.root if root2.converged else None
    return λ1, λ2


def compute_α_star_MaxExp():
    """
    Solve for c such that
    ∫_0^1 1 / (x - x ln x - 1 + 1/c) dx = 1
    """
    def integral(c):
        def integrand(x):
            return 1.0 / (x - x*np.log(x) - 1.0 + 1.0/c)
        val, _ = quad(integrand, 0, 1)
        return val - 1.0

    α_star = root_scalar(integral, bracket=(0.1, 1), method="brentq").root
    return α_star


def save_data(result, filename, header = "α β"):
    arr = np.array(result)
    np.savetxt(filename, arr, fmt="%.12f", header=header)


def read_data(filename):
    arr = np.loadtxt(filename, comments="#")
    α_values = arr[:, 0].tolist()
    β_values = arr[:, 1].tolist()
    return α_values, β_values


def plot_tradeoff_curve(ax, α_values, β_values, mode, color, label=None):
    if mode == "algo":
        ax.plot(α_values, β_values, color=color, label=label)
    elif mode == "hard":
        ax.step(α_values, β_values, where="post", color=color, label=label)
    else:
        raise ValueError(f"Unknown mode {mode!r}. Use 'algo' or 'hard'.")

def plot_tangent():
    return
    '''
        x0, y0 = 0.745, 0
    min_slope = None
    tangent_idx = None
    for i, (x, y) in enumerate(zip(alpha_values, beta_values)):
        if np.isnan(x):
            continue
        slope = (y - y0) / (x - x0) if x != x0 else np.inf
        if min_slope is None or slope < min_slope:
            min_slope = slope
            tangent_idx = i

    if tangent_idx is not None:
        x1, y1 = alpha_values[tangent_idx], beta_values[tangent_idx]
        plt.plot([x0, x1], [y0, y1], '--', label="tangent from (0.745, 0)",color='tab:blue')

    '''


def plot_trivial_algorithm_curve(ax, α_star, color="tab:blue"):
    ax.plot([1 / np.e, α_star], [1 / np.e, 0], color=color, label="Trivial algorithm")


def plot_trivial_hardness_curve(ax, α_star, color="tab:red"):
    ax.plot(
        [1 / np.e, α_star, α_star],
        [1 / np.e, 1 / np.e, 0],
        color=color,
        label="Trivial hardness",
    )


def setup_threshold_plot(ax, λ1, λ2):
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.set_aspect("equal", adjustable="box")

    xticks = [λ1, λ2, 1]
    xtick_labels = [
        f"${val:.3f}$" if i < 2 else f"${val:.0f}$" for i, val in enumerate(xticks)
    ]
    yticks = [1]
    ytick_labels = [f"${val:.0f}$" for val in yticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    ax.text(-0.02, -0.02, r"$0$", ha="right", va="top")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\theta(t)$", labelpad=-5)

    ax.legend(loc=(0.8, 0.8))


def setup_tradeoff_plot_MaxProb(ax, α_star):
    ax.set_xlim(1 / np.e, 0.6)
    ax.set_ylim(0, 0.4)

    xticks = [1 / np.e, α_star]
    xtick_labels = [f"${val:.3f}$" for i, val in enumerate(xticks)]
    yticks = [0, 1 / np.e]
    ytick_labels = [f"${val:.3f}$" if val != 0 else f"${val:.0f}$" for val in yticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.set_xlabel("Consistency (α)")
    ax.set_ylabel("Robustness (β)", labelpad=10, alpha=0)
    ax.legend()
    plt.tight_layout()


def setup_tradeoff_plot_MaxExp(ax, α_star):
    ax.set_xlim(1 / np.e, 0.77)
    ax.set_ylim(0, 0.4)

    xticks = [1 / np.e, α_star]
    xtick_labels = [f"${val:.3f}$" for i, val in enumerate(xticks)]
    yticks = [0, 1 / np.e]
    ytick_labels = [f"${val:.3f}$" if val != 0 else f"${val:.0f}$" for val in yticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.set_xlabel("Consistency (α)")
    ax.set_ylabel("Robustness (β)", labelpad=10)
    ax.legend()
    plt.tight_layout()