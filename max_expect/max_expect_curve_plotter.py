import numpy as np
import matplotlib.pyplot as plt
import os
import re


def optimal_curve_plotter(density=0.005, xaxisrange=(0, 1), yaxisrange=(0, 1)):
    # Load alphas and betas from alpha_betas.txt in the same directory
    file_path = os.path.join(os.path.dirname(__file__), "alpha_betas.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    def _extract(name: str) -> np.ndarray:
        m = re.search(rf"{name}\s*=\s*\[(.*?)\]", content, re.DOTALL)
        if not m:
            raise ValueError(f"{name} array not found in alpha_betas.txt")
        nums = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", m.group(1))
        return np.array([float(x) for x in nums], dtype=float)

    beta_values = _extract("betas")
    alpha_values = _extract("alphas")
    # plot the curve as the boundary of achievable (alpha,beta) pairs, i.e., (x,y:exists i, x<=alpha_i,y<=beta_i)
    plot_alpha_values = []
    plot_beta_values = []
    for i in range(len(alpha_values)):
        if i==0:
            plot_alpha_values.append(alpha_values[i])
            plot_beta_values.append(0)
        plot_alpha_values.append(alpha_values[i])
        plot_beta_values.append(beta_values[i])
        if i!=len(alpha_values)-1:
            plot_alpha_values.append(alpha_values[i+1])
            plot_beta_values.append(beta_values[i])

    plt.plot(plot_alpha_values, plot_beta_values, "-", label="our algorithm", color="tab:green")

    # Find tangent from (0.745, 0) to the curve
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
        plt.plot(
            [x0, x1], [y0, y1], "--", label="tangent from (0.745, 0)", color="tab:blue"
        )

    # trivial algorithm: crossing at (1/e, 1/e) and (0.745,0)
    gamma = 0.745
    new_betas = np.linspace(0, 1 / np.e, 100)
    alpha_values_trivial = [
        gamma + (1 / np.e - gamma) * b / (1 / np.e) for b in new_betas
    ]
    plt.plot(
        alpha_values_trivial,
        new_betas,
        "-",
        label="trivial algorithm",
        color="tab:orange",
    )

    plt.plot(
        [gamma, gamma, 1 / np.e],
        [0, 1 / np.e, 1 / np.e],
        "-",
        label="trivial hardness",
        color="tab:red",
    )

    plt.ylabel("beta")
    plt.xlabel("alpha")
    plt.title("Alpha vs Beta")
    plt.grid(visible=False)
    plt.xlim(xaxisrange)  # Fix x-axis range to [0, 1]
    plt.ylim(yaxisrange)  # Fix y-axis range to [-1, 1]
    plt.legend()


__all__ = ["optimal_curve_plotter"]

if __name__ == "__main__":
    xaxisrange = (0, 0.8)
    yaxisrange = (0, 0.6)
    # Plot the alpha vs beta consistency-robustness curve with its tangent to (0.745, 0),
    # trivial algorithm and trivial hardness
    optimal_curve_plotter(xaxisrange=xaxisrange, yaxisrange=yaxisrange)
    plt.show()