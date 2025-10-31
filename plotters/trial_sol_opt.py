import numpy as np
from scipy.integrate import quad, cumulative_trapezoid
from scipy.special import expi  # Ei(x)
import matplotlib.pyplot as plt
import argparse
from typing import Optional

from max_prob_pareto_curve import compute_lambdas
from opt_analytical_threshold import search_optimal_K,plot_one_K


def C(x: float, lambda1: float, lambda2: float, K: float) -> float:
    """
    C(x) = K/(lambda2 - x) - K/(lambda2 - lambda1) 
    """
    return K  * (1.0 / (lambda2 - x) - 1.0 / (lambda2 - lambda1))


def C_prime(x: float, lambda2: float, K: float) -> float:
    """C'(x) = K * d/dx [1/(lambda2 - x)] = K * (lambda2 - x)^(-2)."""
    return K  * (1.0 / ((lambda2 - x) ** 2))


def compute_inner_terms(v: float, lambda1: float, lambda2: float, K: float, s_eps: float) -> float:
    """
    termA(v) = v * ∫_{t=v}^{lambda2} e^{-t C(v)}/t dt
               + ∫_{s=v}^{lambda2} [ ∫_{t=s}^{lambda2} e^{-t C(s)}/t dt ] ds

    Using exponential integral, ∫ e^{-a t}/t dt = Ei(-a t), hence:
    - First part: v * (Ei(-C(v)*lambda2) - Ei(-C(v)*v))
    - Second part: ∫_{s=v}^{lambda2} (Ei(-C(s)*lambda2) - Ei(-C(s)*s)) ds
    """
    # First part (closed form via Ei)
    Cv = C(v, lambda1, lambda2, K)
    first = v * (expi(-Cv * lambda2) - expi(-Cv * v))

    # Second part: 1D integral over s
    def s_integrand(s):
        Cs = C(s, lambda1, lambda2, K)
        return expi(-Cs * lambda2) - expi(-Cs * s)

    upper = max(v, min(lambda2 - s_eps, lambda2))
    if upper <= v:
        second = 0.0
    else:
        second, _ = quad(s_integrand, v, upper, epsabs=1e-7, epsrel=1e-6, limit=200)

    return first + second


def objective_over_u(lambda1: float, lambda2: float, K: float, n_v: int = 101):
    """
    Build g(v) over a grid and compute N(u) = ∫_{lambda1}^{u} g(v) dv via cumulative trapz.
    Then return arrays (u_grid, J_vals) for the u-grid.
    """
    # Avoid endpoints where expressions may blow up or become 0/0
    eps = 1e-8 * max(1.0, (lambda2 - lambda1))
    s_eps = 1e-8 * max(1.0, (lambda2 - lambda1))

    v_grid = np.linspace(lambda1 + eps, lambda2 - eps, n_v)

    # Precompute g(v) = termA(v) * C'(v)
    g_vals = np.empty_like(v_grid)
    for i, v in enumerate(v_grid):
        termA = compute_inner_terms(v, lambda1, lambda2, K, s_eps)
        g_vals[i] = termA * C_prime(v, lambda2, K)

    # Cumulative integral N(u) across the same grid via trapezoid, starting from lambda1+eps
    # cumulative_trapezoid returns length n-1; prepend 0 at the start to align with v_grid
    N_vals = np.concatenate(([0.0], cumulative_trapezoid(g_vals, v_grid)))

    # Denominator for each u on v_grid
    C_u = K  * (1.0 / (lambda2 - v_grid) - 1.0 / (lambda2 - lambda1))
    denom = 1.0 - np.exp(-C_u)

    # Protect against numerical issues near lambda1 where denom ~ 0
    small = 1e-16
    J_vals = N_vals / np.maximum(denom, small)
    return v_grid, J_vals


def plot_J(beta: float, K: Optional[float] = None, n_v: int = 401, save_path: Optional[str] = None):
    """Plot J(u) over u in [lambda1, lambda2] for given beta and optional K.
    If K is None, use best K from a coarse search. If save_path is given, save instead of show.
    """
    lambda1, lambda2 = compute_lambdas(beta)
    if lambda1 is None or lambda2 is None:
        raise ValueError("Failed to compute (lambda1, lambda2) for the given beta.")

    if K is None:
        best = find_best_K_for_beta(beta)
        K = float(best["best_K"]) if best["best_K"] is not None else 1.0

    u_grid, J_vals = objective_over_u(lambda1, lambda2, K, n_v=n_v)
    idx_min = int(np.nanargmin(J_vals))

    plt.figure(figsize=(7, 4.5))
    plt.plot(u_grid, J_vals, label="J(u)")
    plt.axvline(lambda1, color="gray", linestyle=":", linewidth=1, label=r"$\lambda_1$")
    plt.axvline(lambda2, color="gray", linestyle=":", linewidth=1, label=r"$\lambda_2$")
    plt.plot([u_grid[idx_min]], [J_vals[idx_min]], "ro", label="min J(u)")
    plt.xlabel("u")
    plt.ylabel("J(u)")
    plt.title(f"J(u) for beta={beta:.6g}, K={K:.6g}")
    plt.legend(loc="best")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


def find_best_K_for_beta(
    beta: float,
    K_grid: np.ndarray = None,
    n_v: int = 101,
    ignorance: float = 0.0,
):
    if K_grid is None:
        # Log-spaced K grid to cover a few orders; adjust as needed
        K_grid = np.logspace(-1, 1, 25)  # [0.1, 10]

    lambda1, lambda2 = compute_lambdas(beta)
    if lambda1 is None or lambda2 is None:
        return {
            "beta": beta,
            "lambda1": lambda1,
            "lambda2": lambda2,
            "best_K": np.nan,
            "best_value": np.nan,
            "u_at_min": np.nan,
        }

    best_val = -np.inf
    best_K = None
    best_u_min = None

    for K in K_grid:
        u_grid, J_vals = objective_over_u(lambda1, lambda2, K, n_v=n_v)
        # Apply ignorance cutoff
        if ignorance > 0.0:
            lower_bound = lambda1 + ignorance * (lambda2 - lambda1)
            upper_bound = lambda2 - ignorance * (lambda2 - lambda1)
            valid_indices = np.where((u_grid >= lower_bound) & (u_grid <= upper_bound))[0]
            if len(valid_indices) == 0:
                continue
            J_vals = J_vals[valid_indices]
            u_grid = u_grid[valid_indices]
        # We need min over u, then pick K that maximizes this minimum
        idx_min = int(np.nanargmin(J_vals))
        val = J_vals[idx_min]
        if val > best_val:
            best_val = val
            best_K = K
            best_u_min = u_grid[idx_min]

    return {
        "beta": beta,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "best_K": best_K,
        "best_value": float(best_val),
        "u_at_min": float(best_u_min) if best_u_min is not None else np.nan,
    }

if __name__ == "__main__":
    # trial for solving C(x)=k

    # plot_J(beta=0.344553, K=0.385353, n_v=401)
    betas = np.linspace(10e-6, 1.0 / np.e - 5e-3, 10)


    results = []
    for beta in betas:
        lambda1, lambda2 = compute_lambdas(beta)
        # search the parameter K for function C
        K_low,K_high=search_optimal_K(u0=lambda1,u1=lambda2,K_start=0.1,K_end=2.0,max_iter=20)
        # # test
        # plot_one_K(K_value=K_high, u0=lambda1, u1=lambda2, is_labeled=True, plotter=None, n_points=100)
        

        results.append(res)
    
    # plot alpha,beta
    # plt.plot(results, betas, '-', label="threhsold func with searched K")
    # plt.ylabel('beta')
    # plt.xlabel('alpha')
    # plt.title('Alpha vs Beta')
    plt.grid(visible=False)
    plt.xlim((0,1))
    plt.ylim((0,20))
    plt.legend()
    plt.show()


