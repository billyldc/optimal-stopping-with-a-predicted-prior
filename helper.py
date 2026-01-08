import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.optimize import root_scalar
from scipy.integrate import quad


def lighten_color(color, factor=0.5):
    color_rgb = mcolors.to_rgb(color)
    return mcolors.to_hex([min(c + (1 - c) * factor, 1) for c in color_rgb])


green = "#20BB97"
red = "#EE3F21"
light_green = lighten_color(green, 0.5)
light_red = lighten_color(red, 0.5)
pale_green = lighten_color(green, 0.8)
pale_red = lighten_color(red, 0.8)

x_min = 0.34


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
            return 1.0 / (x - x * np.log(x) - 1.0 + 1.0 / c)

        val, _ = quad(integrand, 0, 1)
        return val - 1.0

    α_star = root_scalar(integral, bracket=(0.1, 1), method="brentq").root
    return α_star


def save_data(result, filename, header="α β"):
    arr = np.array(result)
    np.savetxt(filename, arr, fmt="%.12f", header=header)


def read_data(filename):
    arr = np.loadtxt(filename, comments="#")
    data = arr.tolist()
    return data


def plot_algorithm_curve(ax, α_values, β_values, α_star, label=None):
    ax.plot(α_values, β_values, color=green, linewidth=1, label=label)
    ax.fill_between(α_values, β_values, 0, color=light_green)


def plot_hardness_curve(ax, λ_values, α_values, β_values, α_star, label=None):
    def find_intersection(l1, c1, l2, c2):
        A = [[l1, 1 - l1], [l2, 1 - l2]]
        B = [c1, c2]
        intersection = np.linalg.solve(A, B)
        return tuple(intersection)

    L = λ_values
    C = [
        λ_values[i] * α_values[i] + (1 - λ_values[i]) * β_values[i]
        for i in range(len(λ_values))
    ]
    points = []
    points.append((0, β_values[0]))
    points.append((1 / np.e, β_values[0]))
    for i in range(len(λ_values) - 1):
        points.append(find_intersection(L[i], C[i], L[i + 1], C[i + 1]))
    points.append((α_values[-1], 0))
    x_values, y_values = zip(*points)
    ax.plot(x_values, y_values, color=red, label=label)
    ax.fill_between(x_values, y_values, 1, color=light_red)


def find_tangent(x0, y0, α_values, β_values, mode="robust"):
    """
    Find the tangent point on the curve (α_values, β_values)
    from the base point (x0, y0).
    """
    best_slope = None
    best_idx = None

    if mode == "robust":
        curve = zip(α_values[::-1], β_values[::-1])
        for i, (x, y) in enumerate(curve):
            if np.isnan(x):
                continue
            slope = (y - y0) / (x - x0) if x != x0 else np.inf
            if (best_slope is None and slope != np.inf) or (
                best_slope is not None and slope >= best_slope
            ):
                best_slope = slope
                best_idx = i
        return (α_values[::-1])[best_idx], (β_values[::-1])[best_idx]
    else:
        curve = zip(α_values, β_values)
        for i, (x, y) in enumerate(curve):
            if np.isnan(x):
                continue
            slope = (y - y0) / (x - x0) if x != x0 else np.inf
            if best_slope is None or slope < best_slope:
                best_slope = slope
                best_idx = i
        return α_values[best_idx], β_values[best_idx]


def plot_tangents(ax, α_values, β_values, α_star):
    """
    Plot both tangents of the algorithm curve:
    - from (α_star, 0)
    - from (1/e, 1/e)
    """

    # -----------------------------
    # 1. Tangent from (α_star, 0)
    # -----------------------------
    x0a, y0a = α_star, 0
    x1, y1 = find_tangent(x0a, y0a, α_values, β_values, mode="consistent")
    ax.plot(
        [x0a, x1],
        [y0a, y1],
        color=green,
        linestyle=(0, (3, 2)),
        linewidth=1,
        label="Interpolation with Hill-Kertz / Dynkin",
    )
    ax.fill_between([x0a, x1], [y0a, y1], 0, color=light_green)

    # -----------------------------
    # 2. Tangent from (1/e, 1/e)
    # -----------------------------
    x0b, y0b = 1 / np.e, 1 / np.e
    x2, y2 = find_tangent(x0b, y0b, α_values, β_values, mode="robust")
    ax.plot(
        [x0b, x2],
        [y0b, y2],
        color=green,
        linestyle=(0, (3, 2)),
        linewidth=1,
        # label="Interpolation with Dynkin",
    )
    ax.fill_between([x0b, x2], [y0b, y2], 0, color=light_green)

    return [x1, x2], [y1, y2]


def plot_baseline_algorithm_curve(ax, α_star):
    α = [0, 1 / np.e, α_star]
    β = [1 / np.e, 1 / np.e, 0]
    ax.plot(
        [0, 1 / np.e],
        [1 / np.e, 1 / np.e],
        color=green,
        linewidth=1,
        linestyle=(0, (5, 5)),
        label="Baseline algorithm",
    )
    ax.plot(α, β, color=green, linewidth=1, linestyle=(5, (5, 5)))


def plot_baseline_hardness_curve(ax, α_star):
    α = [0, α_star, α_star]
    β = [1 / np.e, 1 / np.e, 0]
    ax.plot(
        α,
        β,
        color=red,
        linewidth=1,
        linestyle=(0, (5, 5)),
        label="Baseline hardness",
    )


def shade_baseline(ax, α_star):
    α = [0, 1 / np.e, α_star]
    β = [1 / np.e, 1 / np.e, 0]
    ax.fill_between(α, β, 0, color=pale_green)

    α = [0, α_star, α_star, 1]
    β = [1 / np.e, 1 / np.e, 0, 0]
    ax.fill_between(α, β, 1, color=pale_red)


def xbreak_marker(ax, x, size=0.002, lw=0.7, slope=3, gap=0.002):
    """
    Draw a symmetric '//' axis break marker centered at x (data coords).

    Parameters
    ----------
    ax : matplotlib axis
    x : float
        Center of the break marker (data units)
    size : float
        Half-length of each stroke in data units
    lw : float
        Line width
    slope : float
        Vertical rise per unit horizontal (controls angle)
    """

    # First stroke: from (x - size, y0 - slope*size) to (x + size, y0 + slope*size), shifted left by 'gap'
    ax.plot(
        [x - size - gap, x + size - gap],
        [0 - slope * size, 0 + slope * size],
        color="k",
        lw=lw,
        clip_on=False,
    )

    # Second stroke: shifted right by 'gap'
    ax.plot(
        [x - size + gap, x + size + gap],
        [0 - slope * size, 0 + slope * size],
        color="k",
        lw=lw,
        clip_on=False,
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
    ax.grid(True, linestyle="--")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\theta(t)$", labelpad=-5)

    ax.legend(loc=(0.8, 0.8))


def setup_tradeoff_plot_MaxProb(ax, α_star):
    ax.set_xlim(x_min, 0.6)
    ax.set_ylim(0, 0.41)

    xticks = [1 / np.e, α_star]
    xtick_labels = [
        f"${val:.3f}$" if val != 0 else f"${val:.0f}$" for i, val in enumerate(xticks)
    ]
    yticks = [0, 1 / np.e]
    ytick_labels = [f"${val:.3f}$" if val != 0 else f"${val:.0f}$" for val in yticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--")
    xbreak_marker(ax, (x_min + 1 / np.e) / 2)

    ax.set_xlabel("Consistency (α)")
    ax.set_ylabel("Robustness (β)", labelpad=10)
    ax.legend(handlelength=1.7, fontsize=9, loc="lower left")
    plt.tight_layout()


def setup_tradeoff_plot_MaxExp(ax, α_star, x, y):
    ax.set_xlim(x_min, 0.77)
    ax.set_ylim(0, 0.41)
    xticks = [1 / np.e, x[0], α_star]
    xtick_labels = [f"${val:.3f}$" if val != 0 else f"${val:.0f}$" for val in xticks]
    yticks = [0, *y, 1 / np.e]
    ytick_labels = [
        (
            ""
            if 0.01 > np.abs(val - 1 / np.e) > 1e-6
            else ("0" if val == 0 else f"${val:.3f}$")
        )
        for val in yticks
    ]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--")
    xbreak_marker(ax, (x_min + 1 / np.e) / 2)

    ax.set_xlabel("Consistency (α)")
    ax.set_ylabel("Robustness (β)", labelpad=10)
    ax.legend(handlelength=1.7, fontsize=9, loc="lower left")
    plt.tight_layout()
