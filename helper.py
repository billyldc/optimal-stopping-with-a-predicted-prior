import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import root_scalar
from scipy.integrate import quad

green = "#20BB97"
red = "#EE3F21"


def lighten_color(color, factor=0.5):
    color_rgb = mcolors.to_rgb(color)
    return mcolors.to_hex([min(c + (1 - c) * factor, 1) for c in color_rgb])


light_green = lighten_color(green, 0.5)
light_red = lighten_color(red, 0.5)
pale_green = lighten_color(green, 0.8)
pale_red = lighten_color(red, 0.8)


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
    α = [0, 1 / np.e, α_star]
    β = [1 / np.e, 1 / np.e, 0]
    β_interp = np.interp(α_values, α, β)
    ax.fill_between(α_values, β_values, β_interp, color=light_green)


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
    x_idx = np.searchsorted(x_values, α_star, side="right")
    y_idx = np.searchsorted(y_values, 1 / np.e, side="right")
    x_values_limited = x_values[y_idx : x_idx + 1]
    y_values_limited = y_values[y_idx : x_idx + 1]
    y_max = np.maximum(y_values_limited, 1 / np.e)
    ax.fill_between(x_values_limited, y_values_limited, y_max, color=light_red)


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
    Plot both tangents:
    - from (α_star, 0) using minimum slope
    - from (1/e, 1/e) using maximum slope
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
        linestyle=":",
        linewidth=1,
        label="Interpolation with Hill-Kertz",
    )

    # Shade against HK envelope
    α_env = [0, 1 / np.e, α_star]
    β_env = [1 / np.e, 1 / np.e, 0]
    β_interp = np.interp([x0a, x1], α_env, β_env)
    ax.fill_between([x0a, x1], [y0a, y1], β_interp, color=light_green)

    # -----------------------------
    # 2. Tangent from (1/e, 1/e)
    # -----------------------------
    x0b, y0b = 1 / np.e, 1 / np.e
    x2, y2 = find_tangent(x0b, y0b, α_values, β_values, mode="robust")
    ax.plot(
        [x0b, x2],
        [y0b, y2],
        color=green,
        linestyle="--",
        linewidth=1,
        label="Interpolation with Dynkin",
    )

    β_interp2 = np.interp([x0b, x2], α_env, β_env)
    ax.fill_between([x0b, x2], [y0b, y2], β_interp2, color=light_green)
    ax.plot([0, 1 / np.e], [1 / np.e, 1 / np.e], linewidth=1, color=green)

    return [x1, x2], [y1, y2]


# def plot_tangent(ax, α_values, β_values, α_star):
#     x0, y0 = α_star, 0
#     min_slope = None
#     tangent_idx = None
#     for i, (x, y) in enumerate(zip(α_values, β_values)):
#         if np.isnan(x):
#             continue
#         slope = (y - y0) / (x - x0) if x != x0 else np.inf
#         if min_slope is None or slope < min_slope:
#             min_slope = slope
#             tangent_idx = i

#     if tangent_idx is not None:
#         x1, y1 = α_values[tangent_idx], β_values[tangent_idx]
#         ax.plot(
#             [x0, x1],
#             [y0, y1],
#             label="Interpolation with Hill-Kertz",
#             color=green,
#             linestyle=":",
#             linewidth=1,
#         )
#     α = [0, 1 / np.e, α_star]
#     β = [1 / np.e, 1 / np.e, 0]
#     β_interp = np.interp([x0, x1], α, β)
#     ax.fill_between([x0, x1], [y0, y1], β_interp, color=light_green)
#     return x1, y1


def plot_baseline_algorithm_curve(ax, α_star):
    α = [0, 1 / np.e, α_star]
    β = [1 / np.e, 1 / np.e, 0]
    ax.plot(
        α,
        β,
        color=green,
        linewidth=1,
        linestyle=(0, (3, 3)),
        label="Baseline algorithm",
    )
    ax.fill_between(α, β, 0, color=pale_green)


def plot_baseline_hardness_curve(ax, α_star):
    α = [0, α_star, α_star]
    β = [1 / np.e, 1 / np.e, 0]
    ax.plot(
        α,
        β,
        color=red,
        linewidth=1,
        linestyle=(0, (3, 3)),
        label="Baseline hardness",
    )
    α.append(1)
    β.append(0)
    ax.fill_between(α, β, 1, color=pale_red)


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
    ax.set_xlim(0, 0.65)
    ax.set_ylim(0, 0.41)

    xticks = [0, 1 / np.e, α_star]
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

    ax.set_xlabel("Consistency (α)")
    ax.set_ylabel("Robustness (β)", labelpad=10)
    ax.legend()
    plt.tight_layout()


def setup_tradeoff_plot_MaxExp(ax, α_star, x, y):
    ax.set_xlim(0, 0.77)
    ax.set_ylim(0, 0.41)
    print(x)
    print(y)
    xticks = [0, 1 / np.e, x[0], α_star]
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

    ax.set_xlabel("Consistency (α)")
    ax.set_ylabel("Robustness (β)", labelpad=10)
    ax.legend()
    plt.tight_layout()
