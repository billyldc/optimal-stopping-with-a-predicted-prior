import max_prob_algorithm
import max_prob_hardness
from matplotlib import pyplot as plt


if __name__ == "__main__":
    n = 30  # number of arrivals
    density = 0.001  # density: step size for alpha sampling
    strp = "1%x"  # strp: the string representation of the probability mass function for file naming
    m = 1024
    data = max_prob_hardness.alpha_beta_curve(
        n=n,
        m=m,
        simplify=True,
        density=density,
        load_from_file=True,
        suppfunc=lambda x: 1 / x,
        strp=strp,
    )
    xaxisrange = (0, 0.8)
    yaxisrange = (0, 0.6)
    # Plot the consistency-robustness curve of our algorithm \mathcal{A}
    max_prob_algorithm.optimal_curve_plotter(
        density=0.0001, xaxisrange=xaxisrange, yaxisrange=yaxisrange
    )
    # Plot the constructed hardness curve against the trivial hardness curve
    data.plot(xaxisrange=xaxisrange, yaxisrange=yaxisrange, add_legend=True)
    plt.show()
