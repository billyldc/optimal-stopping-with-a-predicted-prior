import numpy as np
from math import factorial
from scipy.integrate import quad
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import os
from helper import compute_λ, save_data, read_data, plot_algorithm_curve


def solve_γ():
    func = lambda γ: sum(γ**j / (factorial(j) * j) for j in range(1, 100)) - 1
    return root_scalar(func, bracket=(0.1, 1), method="brentq").root


def compute_α_for_MaxProb(β, γ):
    # Compute β + \int_{λ1}^{λ2} \int_s^1 exp(-γ * t / (1-s)) / t dt ds
    λ1, λ2 = compute_λ(β)

    def inner_integrand(t, s):
        return np.exp(-γ * t / (1.0 - s)) / t

    def inner_integral(s):
        t_lower = s
        t_upper = 1.0
        val, _ = quad(inner_integrand, t_lower, t_upper, args=(s,))
        return val

    double_integral, _ = quad(lambda s: inner_integral(s), λ1, λ2)

    return β + double_integral


def plot_algorithm_MaxProb(ax, density=0.0001, label=None, filename=None):
    if not os.path.exists(filename):
        γ = solve_γ()
        β_values = np.linspace(0, 1 / np.e, int(1 / np.e / density) + 2)
        α_values = []
        for β in β_values:
            try:
                α_values.append(compute_α_for_MaxProb(β, γ))
            except Exception:
                print(f"Failed to compute α for β={β}")
                α_values.append(np.nan)
                continue
        α_values = np.append(α_values, 0)
        β_values = np.append(β_values, 1 / np.e)
        arr = np.column_stack((α_values, β_values))
        save_data(arr, filename)
    else:
        data = read_data(filename)
        α_values = [row[0] for row in data]
        β_values = [row[1] for row in data]
    plot_algorithm_curve(ax, α_values, β_values, max(α_values), label=label)


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=500)
    plot_algorithm_MaxProb(ax, filename="Algorithm_MaxProb.txt")
    plt.tight_layout()
    plt.show()
