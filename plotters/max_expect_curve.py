import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from max_prob_pareto_curve import f_lambda, compute_lambdas, compute_alpha, solve_c
from opt_analytical_threshold import optimal_alpha_beta

import matplotlib.pyplot as plt

c= solve_c()
print("Solved c:", c)
def compute_alpha(beta, lambda1, lambda2):
    # \alpha(\beta) = \beta +\int_{\lambda_1}^1\frac{\min \{t,\lambda_2\}-\lambda_1}{t} e^{-t} ~ \dif t.
    if lambda1 is None or lambda2 is None:
        return np.nan

    def integrand(t):
        # handle the removable singularity at t=0 when lambda1 == 0
        if lambda1 == 0 and t < 1e-12:
            return np.exp(-t)  # limit is e^{-t} because min(t,lambda2)/t -> 1 for small t
        numerator = min(t, lambda2) - lambda1
        return (numerator / t) * np.exp(-t)

    t_lower = max(lambda1, 0.0)
    t_upper = 1.0
    if t_lower >= t_upper:
        return beta

    try:
        val, err = quad(integrand, t_lower, t_upper, epsabs=1e-9, epsrel=1e-9, limit=200)
    except Exception:
        return np.nan

    return beta + val

def optimal_curve_plotter(density=0.005,xaxisrange=(0,1),yaxisrange=(0,1)):
    # beta_values = np.arange(1e-9, 1.0/np.e, density)
    # alpha_values = []
    # for beta in beta_values:
    #     try:
    #         lambda1, lambda2 = compute_lambdas(beta)
    #     except Exception:
    #         alpha_values.append(np.nan)
    #         continue
    #     alpha_values.append(compute_alpha(beta, lambda1, lambda2))
    # our algorithm
    beta_values, alpha_values = optimal_alpha_beta(density=density)
    plt.plot(alpha_values, beta_values, '-', label="our algorithm",color='tab:blue')

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
        plt.plot([x0, x1], [y0, y1], '--', label="tangent from (0.745, 0)",color='tab:blue')

    # trivial algorithm: crossing at (1/e, 1/e) and (0.745,0)
    gamma=0.745
    new_betas=np.linspace(0,1/np.e,100)
    alpha_values_trivial = [gamma + (1/np.e - gamma) * b / (1/np.e) for b in new_betas]
    plt.plot(alpha_values_trivial, new_betas, '-', label="trivial algorithm",color='tab:orange')

    plt.plot([gamma, gamma, 1/np.e], [0, 1/np.e, 1/np.e], '-', label="trivial hardness",color='tab:red')

    plt.ylabel('beta')
    plt.xlabel('alpha')
    plt.title('Alpha vs Beta')
    plt.grid(visible=False)
    plt.xlim(xaxisrange)  # Fix x-axis range to [0, 1] 
    plt.ylim(yaxisrange)  # Fix y-axis range to [-1, 1]
    plt.legend()

__all__ = ["optimal_curve_plotter"]

if __name__ == '__main__':
    xaxisrange = (0, 0.8)
    yaxisrange = (0, 0.6)
    optimal_curve_plotter(xaxisrange=xaxisrange,yaxisrange=yaxisrange)
    plt.show()