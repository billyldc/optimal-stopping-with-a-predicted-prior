import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar

import matplotlib.pyplot as plt


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


# solve c as the root of \sum_{j=1}^{\infty} \frac{c^j}{j!j} = 1
def solve_c():
    func = lambda c: sum(c**j / (np.math.factorial(j) * j) for j in range(1, 100)) - 1
    root = root_scalar(func, bracket=(0.1, 10), method="brentq")
    return root.root if root.converged else None


# Print the solved c value
c = solve_c()
print("Solved c:", c)


def compute_alpha(beta, lambda1, lambda2):
    # Compute beta + \int_{lambda1}^{lambda2} \int_s^1 exp(-c * t / (1-s)) / t dt ds
    if lambda1 is None or lambda2 is None:
        return np.nan

    eps = 1e-12
    s_lower = max(lambda1, eps)
    s_upper = min(lambda2, 1.0 - eps)

    if s_upper <= s_lower:
        double_integral = 0.0
    else:

        def inner_integrand(t, s):
            return np.exp(-c * t / (1.0 - s)) / t

        def inner_integral(s):
            t_lower = s
            t_upper = 1.0 - eps
            val, _ = quad(
                inner_integrand,
                t_lower,
                t_upper,
                args=(s,),
                epsabs=1e-9,
                epsrel=1e-9,
                limit=200,
            )
            return val

        double_integral, _ = quad(
            lambda s: inner_integral(s),
            s_lower,
            s_upper,
            epsabs=1e-9,
            epsrel=1e-9,
            limit=200,
        )

    return beta + double_integral


def optimal_curve_plotter(density=0.0001, xaxisrange=(0, 1), yaxisrange=(0, 1)):
    """Plot the optimal alpha vs beta consistency curve against the trivial algorithm."""
    beta_values = np.arange(1e-9, 1.0 / np.e, density)
    alpha_values = []
    for beta in beta_values:
        try:
            lambda1, lambda2 = compute_lambdas(beta)
        except Exception:
            alpha_values.append(np.nan)
            continue
        alpha_values.append(compute_alpha(beta, lambda1, lambda2))

    # our algorithm
    plt.plot(alpha_values, beta_values, "-", label="our algorithm")
    # trivial algorithm: crossing at (1/e, 1/e) and (0.58,0)
    alpha_values = [0.58 + (1 / np.e - 0.58) * b / (1 / np.e) for b in beta_values]
    plt.plot(alpha_values, beta_values, "-", label="trivial algorithm")
    plt.ylabel("beta")
    plt.xlabel("alpha")
    plt.title("Alpha vs Beta")
    plt.grid(False)
    plt.xlim(xaxisrange)  # Fix x-axis range to [0, 1]
    plt.ylim(yaxisrange)  # Fix y-axis range to [-1, 1]


__all__ = ["optimal_curve_plotter"]

if __name__ == "__main__":
    # Example usage and testing
    optimal_curve_plotter()
    plt.show()
