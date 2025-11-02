import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar

import matplotlib.pyplot as plt


def solve_and_plot(n=10):
    """
    For a given n, solve the equation and plot a as a function of s.

    Equation: ∫[s, 1] ( ((1-t)/a + t)^(n-1) - 1 ) / (1-t) dt - 1 = 0
    """

    # Define the integrand
    def integrand(t, a, n_val):
        base = (1 - t) / a + t
        return (np.power(base, n_val - 1) - 1) / (1 - t)

    # Define the function whose root we want to find: F(a) = integral - 1
    def equation_to_solve(a, s_val, n_val):
        # The valid range for a is (0, inf), in particular a > 0
        if a <= 0:
            return np.inf

        # Use quad for numerical integration
        integral_val, error = quad(integrand, s_val, 1, args=(a, n_val))
        return integral_val - 1

    # Set the range for s
    # Start from a value slightly greater than 0 to avoid numerical issues when s=0 and a->0
    s_values = np.linspace(0, 1 - 1e-9, 1000)
    a_solutions = []

    # Initial guess value
    a_guess = 0.5

    # Iterate over each s value and solve for the corresponding a
    for s in s_values:
        try:
            # Use root_scalar to find the root
            # bracket defines the interval where the root is located
            sol = root_scalar(
                equation_to_solve,
                args=(s, n),
                bracket=[1e-9, 5],
                method="brentq",
                x0=a_guess,
            )
            if sol.converged:
                a_solutions.append(sol.root)
                a_guess = (
                    sol.root
                )  # Use the previous solution as the initial guess for the next
            else:
                a_solutions.append(np.nan)
        except (ValueError, RuntimeError):
            # If solving fails, record as nan
            a_solutions.append(np.nan)

    # Create two side-by-side plots: left = GM only, right = GM + Dynkin
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))

    # Common settings
    xlims = (-0.1, 1.1)
    ylims = (-0.1, 1.1)

    # Left: original two threshold-based algorithms
    ax = ax_left
    ax.plot(
        s_values,
        a_solutions,
        color="blue",
        linewidth=2.5,
        linestyle="-",
        label=f"Gilbert-Mosteller (n={n})",
    )

    # Dynkin: avoid connecting across the discontinuity at s = 1/e by plotting two segments
    cutoff = 1 / np.e
    mask_left = s_values < cutoff
    mask_right = s_values > cutoff

    if mask_left.any():
        ax.plot(
            s_values[mask_left],
            np.ones(int(mask_left.sum())),
            color="purple",
            linewidth=2.5,
            linestyle="-",
        )
    if mask_right.any():
        ax.plot(
            s_values[mask_right],
            np.zeros(int(mask_right.sum())),
            color="purple",
            linewidth=2.5,
            linestyle="-",
            label=f"Dynkin's (n={n})",
        )
    # Add dashed vertical line at x = 1/e from y=0 to 1
    x_cut = 1 / np.e
    ax.plot([x_cut, x_cut], [0, 1], color="purple", linestyle="--", linewidth=2)

    ax.set_ylabel("value")
    ax.set_xlabel("t")
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_position(("data", 0.0))
    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(1.5)
    ax.legend()
    # ax.set_title('original two threshold-based algorithms')

    # Left: parameterized by lambda_1, lambda_2
    def f_lambda(x):
        # Compute lambda1 and lambda2 for beta values from 0 to 1/e
        if x <= 0 or x >= 1:
            return float("inf")
        return -x * np.log(x)

    def compute_lambdas(beta):
        if beta < 0 or beta > 1 / np.e:
            raise ValueError("Beta must be in the range [0, 1/e]")
        if beta == 0:
            return 0, 1
        # Small root (lambda1) in interval (0, 1/e)
        root1 = root_scalar(
            lambda x: f_lambda(x) - beta, bracket=(1e-10, 1 / np.e), method="brentq"
        )
        lambda1 = root1.root if root1.converged else None

        # Large root (lambda2) in interval (1/e, 1)
        root2 = root_scalar(
            lambda x: f_lambda(x) - beta, bracket=(1 / np.e, 1 - 1e-10), method="brentq"
        )
        lambda2 = root2.root if root2.converged else None
        return lambda1, lambda2

    lambda_1, lambda_2 = compute_lambdas(1 / 3)

    ax = ax_right

    # Prepare defaults for vertical line endpoints
    left_end = 1
    right_end = 0

    # Segment 1: s < lambda_1, a = 1 (no label here)
    mask_left = s_values < lambda_1
    if mask_left.any():
        ax.plot(
            s_values[mask_left],
            np.ones(int(mask_left.sum())),
            color="red",
            linewidth=2.5,
            linestyle="-",
        )

    # Segment 2: lambda_1 <= s <= lambda_2, a follows GM solution (single label kept here)
    mask_middle = (s_values >= lambda_1) & (s_values <= lambda_2)
    if mask_middle.any():
        mid_solutions = np.array(a_solutions)[mask_middle]
        left_end, right_end = mid_solutions[0], mid_solutions[-1]
        ax.plot(
            s_values[mask_middle],
            mid_solutions,
            color="red",
            linewidth=2.5,
            linestyle="-",
            label="Modified Algorithm (piecewise)",
        )

    # Segment 3: s > lambda_2, a = 0 (no label here)
    mask_right = s_values > lambda_2
    if mask_right.any():
        ax.plot(
            s_values[mask_right],
            np.zeros(int(mask_right.sum())),
            color="red",
            linewidth=2.5,
            linestyle="-",
        )

    # Add dashed vertical lines at discontinuity points (keep their labels)
    ax.plot(
        [lambda_1, lambda_1], [left_end, 1], color="red", linestyle="--", linewidth=2
    )
    ax.plot(
        [lambda_2, lambda_2], [0, right_end], color="red", linestyle="--", linewidth=2
    )

    ax.set_ylabel("value")
    ax.set_xlabel("t")
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_position(("data", 0.0))
    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(1.5)
    ax.legend()
    # ax.set_title(f'Threshold-based Algorithm with n={n}')

    plt.tight_layout()
    plt.savefig(f"max_prob_n{n}.png", dpi=300)


if __name__ == "__main__":
    solve_and_plot(n=10)
