import numpy as np
import matplotlib.pyplot as plt
from Algorithm_MaxProb import plot_algorithm_MaxProb, solve_γ, compute_α_for_MaxProb
from Hardness_MaxProb import plot_hardness_MaxProb
from helper import (
    plot_baseline_algorithm_curve,
    plot_baseline_hardness_curve,
    shade_baseline,
    setup_tradeoff_plot_MaxProb,
)


def plot_tradeoff_MaxProb(n, K, algorithm_filename=None, hardness_filename=None):

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=1000)

    γ = solve_γ()
    α_star = compute_α_for_MaxProb(0, γ)

    plot_baseline_algorithm_curve(ax, α_star)
    plot_baseline_hardness_curve(ax, α_star)
    plot_algorithm_MaxProb(ax, label="Theorem 2", filename=algorithm_filename)
    plot_hardness_MaxProb(
        ax, n, K, α_star, label="Theorem 3", filename=hardness_filename
    )
    shade_baseline(ax, α_star)
    setup_tradeoff_plot_MaxProb(
        ax,
        α_star,
    )
    plt.show()


if __name__ == "__main__":

    n, K = 30, 1024
    plot_tradeoff_MaxProb(
        n, K, "Algorithm_MaxProb.txt", f"Hardness_MaxProb_n={n}_K={K}.txt"
    )
