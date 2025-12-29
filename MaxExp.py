import numpy as np
import matplotlib.pyplot as plt
from helper import (
    lighten_color,
    compute_α_star_MaxExp,
    read_data,
    plot_algorithm_curve,
    plot_tangents,
    plot_baseline_algorithm_curve,
    plot_baseline_hardness_curve,
    setup_tradeoff_plot_MaxExp,
)


green = "#20BB97"
red = "#EE3F21"
light_green = lighten_color(green, 0.5)
light_red = lighten_color(red, 0.5)
pale_green = lighten_color(green, 0.8)
pale_red = lighten_color(red, 0.8)


def plot_tradeoff_MaxExp(filename=None):

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=1000)

    α_star = compute_α_star_MaxExp()

    plot_baseline_algorithm_curve(ax, α_star)
    plot_baseline_hardness_curve(ax, α_star)

    data = read_data(filename)
    α_values = [row[0] for row in data]
    β_values = [row[1] for row in data]

    # add a small offset to α_values for better visualization
    α_values = np.insert(α_values, 0, α_values[0])
    β_values = np.insert(β_values, 0, 0.0)

    α_cut = 1 / np.e
    β_cut = np.interp(α_cut, α_values[::-1], β_values[::-1])
    mask = α_values >= α_cut
    α_trunc = np.append(α_values[mask], α_cut)
    β_trunc = np.append(β_values[mask], β_cut)

    plot_algorithm_curve(ax, α_trunc, β_trunc, α_star, label="Theorem 1")
    x, y = plot_tangents(ax, α_trunc, β_trunc, α_star)

    α = [0, 1 / np.e, α_star]
    β = [1 / np.e, 1 / np.e, 0]
    ax.fill_between(α, β, 0, color=pale_green)

    α = [0, α_star, α_star, 1]
    β = [1 / np.e, 1 / np.e, 0, 0]
    ax.fill_between(α, β, 1, color=pale_red)

    setup_tradeoff_plot_MaxExp(ax, α_star, x, y)
    plt.show()


if __name__ == "__main__":

    plot_tradeoff_MaxExp("Algorithm_MaxExp.txt")
