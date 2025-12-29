import numpy as np
import matplotlib.pyplot as plt
from helper import (
    compute_α_star_MaxExp,
    read_data,
    plot_algorithm_curve,
    plot_tangent,
    plot_baseline_algorithm_curve,
    plot_baseline_hardness_curve,
    setup_tradeoff_plot_MaxExp,
)


def plot_tradeoff_MaxExp(filename=None):

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=1000)

    α_star = compute_α_star_MaxExp()

    plot_baseline_algorithm_curve(ax, α_star)
    plot_baseline_hardness_curve(ax, α_star)
    
    data = read_data(filename)
    α_values = [row[0] for row in data]
    β_values = [row[1] for row in data]
    # add a small offset to α_values for better visualization
    α_values = np.insert(α_values, 0, α_values[0] )
    β_values = np.insert(β_values, 0, 0.0)
    
    plot_algorithm_curve(ax, α_values, β_values, α_star, label="Theorem 1")
    x, y = plot_tangent(ax, α_values, β_values, α_star)
    
    setup_tradeoff_plot_MaxExp(ax, α_star, x, y)
    plt.show()


if __name__ == "__main__":

    plot_tradeoff_MaxExp("Algorithm_MaxExp.txt")
