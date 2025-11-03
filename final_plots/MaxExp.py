import numpy as np
import matplotlib.pyplot as plt
from helper import (
    compute_α_star_MaxExp,
    read_data,
    plot_tradeoff_curve,
    plot_tangent,
    plot_trivial_algorithm_curve,
    plot_trivial_hardness_curve,
    setup_tradeoff_plot_MaxExp,
)


def plot_tradeoff_MaxExp(filename=None):

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=1000)

    α_star = compute_α_star_MaxExp()

    plot_trivial_algorithm_curve(ax, α_star)
    α_values, β_values = read_data(filename)
    plot_tradeoff_curve(ax, α_values, β_values, "algo", color = "tab:green", label="Our algorithm")
    plot_tangent()
    plot_trivial_hardness_curve(ax, α_star)

    setup_tradeoff_plot_MaxExp(ax, α_star)
    plt.show()


if __name__ == "__main__":

    plot_tradeoff_MaxExp("Algorithm_MaxExp.txt")
