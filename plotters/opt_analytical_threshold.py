import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from scipy.special import expi

# dC/du from implicit ODE:
# -exp(-u C) + (u C'/C)(exp(-C) - exp(-u C)) = -K C' exp(-C)
# => C' = C e^{-u C} / [ u (e^{-C} - e^{-u C}) + K C e^{-C} ]

def modified_compute_min_ratio(alpha, lambda2, C, z_max=200.0, grid_points=200, quad_eps=1e-9, if_plot=False, if_output_process=False):
    """
    Find z (in [C, z_max]) solving
      G(z) :=l \int_{l}^{1}\frac{e^{\left(1-t\right)z}}{t}dt-l\int_{l}^{1}\frac{e^{-tz}}{t}dt+e^{-lz}-le^{-z}-lz\int_{l}^{1}\frac{e^{-tz}}{t}dt-\alpha\left(1-e^{-c}\right)-e^{-lc}+le^{-c}+cl\int_{l}^{1}\frac{e^{-tc}}{t}dt
    and then return (z, F(z)) where F(z) = lambda2 * ∫_{lambda2}^1 e^{-t z}/t dt.

    Returns (None, np.nan) on failure.
    """
    if C < 0:
        raise ValueError("C must be non-negative")
    if not (0.0 < lambda2 < 1.0):
        raise ValueError("lambda2 must be in (0,1)")

    def integral_Ei(a: float) -> float:
        #L=lambda2
        # I(a) = ∫_{L}^{1} e^{-a t}/t dt = Ei(-a) - Ei(-L a)
        a = float(a)
        return float(expi(-a) - expi(-lambda2 * a))

    def integral_I(z):
        # I(z) = \int_{lambda2}^1 (e^{-t C} - exp(-t z)) / t^2 dt
        try:
            val, _ = quad(lambda t: (np.exp(-t * C) - np.exp(-t * z)) / (t * t),
                          lambda2, 1.0, epsabs=quad_eps, epsrel=quad_eps, limit=200)
            return val
        except Exception:
            return float('nan')

    def H(z):
        # H(z)=(e^{w}-1) \lambda_2 \int_{\lambda_2}^{1} \frac{e^{-t w}}{t} \dd t - \alpha_1^*(\lambda_1,\lambda_2) (1-e^{-C^*(\lambda_2)}) - \lambda_2 \int_{\lambda_2}^{1} \frac{e^{-t C^*(\lambda_2)} - e^{-t w}}{t^2} \dd t
        term_1= (np.exp(z)-1.0) * lambda2 * integral_Ei(z)
        const_term = alpha * (1.0 - np.exp(-C))
        term_3= - lambda2 * integral_I(z)
        return term_1 - const_term + term_3

    if if_plot:
        try:
            zs_plot = np.linspace(0, 1000, max(300, int(grid_points)))
            Gs_plot = np.array([H(zv) for zv in zs_plot])
            mask = np.isfinite(Gs_plot)
            if mask.any():
                plt.figure(figsize=(8, 5))
                plt.plot(zs_plot[mask], Gs_plot[mask], label='G(z)')
                plt.axhline(0.0, color='k', linewidth=0.8)
                plt.xlabel('z')
                plt.ylabel('G(z)')
                plt.title('Equation G(z)=0')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show()
        except Exception:
            pass

    # use binary search to find root of G(z)=0 over [c, inf)
    z_min=C
    z_max=1e8
    iterations=50

    if if_output_process:
        print(H(z_min))
        print(H(z_max))
    if H(C)>=0:
        return C, alpha
    if H(z_max)<0:
        if if_output_process:
            print("Warning: G(z_max)<0; no root found in [C, z_max]")
        return None, np.nan
    for _ in range(iterations):
        mid=(z_min+z_max)/2
        H_mid=H(mid)
        if if_output_process:
            print(f"The mid point is at H({mid})={H_mid}")
        if H_mid>0 or np.isnan(H_mid):
            z_max=mid
        else:
            z_min=mid
    z_root=0.5*(z_min+z_max)
    print(f"Modified compute_min_ratio: found root z={z_root} with H(z)={H(z_root)} after {iterations} iterations.")

    # N(w^*)=e^w \lambda_2 \int_{\lambda_2}^{1} \frac{e^{-t w}}{t} \dd t
    out_F=np.exp(z_root) * lambda2 * integral_Ei(z_root)
    return z_root, out_F


def compute_min_ratio(alpha, lambda2, C, z_max=200.0, grid_points=200, quad_eps=1e-9, if_plot=False):
    """
    Compute min_{z in [C, inf)} F(z) where
      F(z) = ( alpha*(1 - e^{-C}) + lambda2 * \int_{lambda2}^1 (e^{-t C} - e^{-t z}) / t^2 dt ) / (1 - e^{-z})

    Parameters
    - alpha: scalar
    - lambda2: integration lower limit, must satisfy 0 < lambda2 < 1
    - C: scalar used in the prefactor alpha*(1-e^{-C}) and in the integrand
    - z_max: finite upper bound used for bounded minimization (default 200)
    - grid_points: number of initial samples for bracket selection
    - quad_eps: absolute/relative tolerance passed to scipy.integrate.quad
    - if_plot: if True, plot F(z) over [C, z_max] and mark the minimizer

    Returns (z_opt, F_opt). Returns (None, np.nan) on failure.
    """

    if C < 0:
        raise ValueError("C must be non-negative")
    if not (0.0 < lambda2 < 1.0):
        raise ValueError("lambda2 must be in (0,1)")

    # constant part
    const_term = alpha * (1.0 - np.exp(-C))

    def integral_I(z):
        # I(z) = \int_{lambda2}^1 (e^{-t C} - exp(-t z)) / t^2 dt
        try:
            val, _ = quad(lambda t: (np.exp(-t * C) - np.exp(-t * z)) / (t * t),
                          lambda2, 1.0, epsabs=quad_eps, epsrel=quad_eps, limit=200)
            return val
        except Exception:
            return float('nan')

    def F(z):
        # domain: z >= C and z > 0
        if not np.isfinite(z) or z <= 0 or z < C:
            return float('inf')
        I = integral_I(z)
        if not np.isfinite(I):
            return float('inf')
        denom = 1.0 - np.exp(-z)
        if denom <= 0:
            # numerical safeguard (shouldn't happen for z>0)
            return float('inf')
        return (const_term + lambda2 * I) / denom

    # Determine a practical finite upper bound for z
    if z_max <= C:
        z_max = C + 50.0
    else:
        z_max = max(z_max, C + 50.0)

    # coarse grid search to find a good starting bracket
    zs = np.linspace(C, z_max, grid_points)
    Fs = np.array([F(zv) for zv in zs])
    finite_mask = np.isfinite(Fs)
    if not finite_mask.any():
        return None, np.nan

    # pick best finite start
    idx_best = np.nanargmin(Fs)
    z_best = float(zs[idx_best])

    # refine by direct traversal on a finer grid (no bounded scalar minimizer)
    try:
        fine_points = max(1000, grid_points * 10)
        zs_fine = np.linspace(C, z_max, fine_points)
        Fs_fine = np.array([F(zv) for zv in zs_fine])
        finite_mask_fine = np.isfinite(Fs_fine)
        if not finite_mask_fine.any():
            # fallback to coarse best
            z_opt, F_opt = z_best, float(F(z_best))
        else:
            idx_opt = int(np.nanargmin(Fs_fine))
            z_opt, F_opt = float(zs_fine[idx_opt]), float(Fs_fine[idx_opt])
    except Exception:
        z_opt, F_opt = z_best, float(F(z_best))

    # Optional plotting of F(z)
    if if_plot:
        try:
            zs_plot = np.linspace(C, z_max, max(300, grid_points))
            Fs_plot = np.array([F(zv) for zv in zs_plot])
            mask = np.isfinite(Fs_plot)
            if mask.any():
                plt.figure(figsize=(8, 5))
                plt.plot(zs_plot[mask], Fs_plot[mask], label='F(z)')
                plt.scatter([z_opt], [F_opt], color='red', zorder=5, label=f'opt z={z_opt:.6g}')
                plt.axvline(z_opt, color='red', linestyle='--', linewidth=0.8)
                plt.xlabel('z')
                plt.ylabel('F(z)')
                plt.title('Objective F(z) and minimizer')
                plt.grid(True)
                plt.legend()
                plt.ylim(0,1)
                plt.tight_layout()
                plt.show()
        except Exception:
            # plotting must not break the computation
            pass

    return z_opt, F_opt

def denom(u, C, K):
    e_uc = np.exp(-u * C)
    e_c = np.exp(-C)
    return u * (e_c - e_uc) + K * C * e_c


def dC_du(u, y, K):
    C = y[0]
    # Safeguards for small C and near-singular denominators
    C_small = 1e-10
    D_small = 1e-10
    if abs(C) < C_small:
        # For small C, C' ~ 1 / (K + u(u-1)) and as u->0, ~ 1/K
        denom_approx = K + u * (u - 1.0)
        denom_approx = np.sign(denom_approx) * max(abs(denom_approx), D_small)
        return [1.0 / denom_approx]

    e_uc = np.exp(-u * C)
    num = C * e_uc
    den = denom(u, C, K)

    if abs(den) < D_small:
        # Fall back to small-C-style approximation if denominator nearly vanishes
        denom_approx = K + u * (u - 1.0)
        denom_approx = np.sign(denom_approx) * max(abs(denom_approx), D_small)
        return [1.0 / denom_approx]

    return [num / den]


# def make_events(K, C_max=1e3, den_thresh=1e-10, slope_max=1e6):
#     def ev_cmax(u, y):
#         return C_max - abs(y[0])
#     ev_cmax.terminal = True
#     ev_cmax.direction = -1

#     def ev_den(u, y):
#         d = denom(u, y[0], K)
#         return abs(d) - den_thresh
#     ev_den.terminal = True
#     ev_den.direction = -1

#     def ev_slope(u, y):
#         C = y[0]
#         e_uc = np.exp(-u * C)
#         e_c = np.exp(-C)
#         num = C * e_uc
#         den = u * (e_c - e_uc) + K * C * e_c
#         val = np.inf if den == 0 else abs(num / den)
#         return slope_max - val
#     ev_slope.terminal = True
#     ev_slope.direction = -1

#     return [ev_cmax, ev_den, ev_slope]

def modify_C(lambda_2, C):
    # compute lambda_2 * \int_{lambda_2}^1 e^{danC(1-t)} / t dt
    # If C is None (no root found) or lambda_2 is out of (0,1) range, return NaN or 0 as appropriate.
    if C is None:
        # no valid C to modify; caller will handle NaN
        return np.nan

    if lambda_2 >= 1.0:
        # integral over empty interval
        return 0.0

    if lambda_2 <= 0.0:
        # integral singular at 0; cannot compute reliably here
        return np.nan

    def integrand(t):
        # safe integrand assuming t in (lambda_2, 1]
        return np.exp(C * (1 - t)) / t

    try:
        val, err = quad(integrand, lambda_2, 1.0, epsabs=1e-9, epsrel=1e-9, limit=200)
        return lambda_2 * val
    except Exception:
        return np.nan

def compute_limit_C(lambda_2,alpha):
    # compute C such that alpha=lambda_2 * (e^{C(1-lambda_2)} -1)/C
    if lambda_2==1:
        return 0.7006
    def func(C):
        if C==0:
            return  lambda_2*(1-lambda_2)-alpha
        return lambda_2 * (np.exp(C * (1 - lambda_2)) - 1) / C - alpha
    print(func(1e-10),func(1000.0))
    try:
        root = root_scalar(func, bracket=(1e-10, 1000.0), method='brentq')
    except Exception:
        return None
    return root.root if root.converged else None

def solve_C_for_K(K, u0=0.0, u1=1.0, method='Radau', rtol=1e-10, atol=1e-10,n_points=200):
    """
    Solve the ODE for given K and return (sol, reached_full_u)
    reached_full_u is True when the solver produced values at all points in u_eval.
    On exception, returns (None, False).
    """
    C0=0
    u_eval = np.linspace(u0, u1, n_points)
    # events = make_events(K)
    try:
        sol = solve_ivp(
            lambda u, y: dC_du(u, y, K),
            (u0, u1),
            [C0],
            t_eval=u_eval,
            method=method,
            rtol=rtol,
            atol=atol,
            # events=events,
        )
        reached_full = sol.t.size == u_eval.size
        reached_full=reached_full and min(sol.y[0])>=0
        return sol, reached_full
    except Exception:
        return None, False

def search_optimal_K(u0=0.0, u1=1.0, K_start=0.01, K_end=1.0, max_iter = 10,n_points=5000):
    """
    Binary search for the boundary K:
    find largest K_low such that solver does NOT reach u1 (sol.t.size < len(u_eval))
    and smallest K_high such that solver DOES reach u1 (sol.t.size == len(u_eval)).
    Returns (K_low, K_high) or (None, None) if no valid bracket found.
    """

    def reached_full_u(K):
        sol, reached = solve_C_for_K(K, u0=u0, u1=u1,n_points=n_points)
        return bool(reached)

    # Ensure we have a bracket: K_start should fail, K_end should succeed
    start_reached = reached_full_u(K_start)
    end_reached = reached_full_u(K_end)

    if start_reached:
        print("K_start already reaches u1; no failing K in given range.")
        return (None, None)
    if not end_reached:
        print("K_end does not reach u1; no reaching K in given range.")
        return (None, None)

    K_low, K_high = float(K_start), float(K_end)
    tol = 1e-10
    it = 0
    while (K_high - K_low) > tol and it < max_iter:
        K_mid = 0.5 * (K_low + K_high)
        if reached_full_u(K_mid):
            K_high = K_mid
        else:
            K_low = K_mid
        it += 1

    print(f"Search done in {it} iters. Largest failing K ~ {K_low}, smallest reaching K ~ {K_high}")
    return (K_low, K_high)

def plot_one_K(K_value, u0=0.0, u1=1.0, is_labeled=True, plotter=None, n_points=100,rounding=10):
    ## plot for one K_value and u0,u1
    ## label the curve if required
    try:
        sol, reached_full = solve_C_for_K(K_value, u0=u0, u1=u1, n_points=n_points)
        if sol is None:
            print(f"No solution (exception) for K={K_value}")

        # Always plot computed portion, even if not successful
        u_plot = np.concatenate(([0.0], sol.t)) if sol.t.size else np.array([0.0])
        c_plot = np.concatenate(([0.0], sol.y[0])) if sol.t.size else np.array([0.0])
        # Filter out any non-finite values
        msk = np.isfinite(u_plot) & np.isfinite(c_plot)
        u_plot, c_plot = u_plot[msk], c_plot[msk]
        if is_labeled:
            label = f"u0={u0}, u1={u1}, K={K_value:.{rounding}f}"
            plt.plot(u_plot, c_plot, label=label)
        else:
            plt.plot(u_plot, c_plot)
    except Exception as e:
        print(f"Error for K={K_value}: {e}")
    return plotter

def plot_batch(K_u_label_tuples,ylimit=(0,20),num_points=100,rounding=10):
    ## plot for a batch of (K,u0,u1) tuples
    plotter=plt.figure(figsize=(10, 6))
    for K,u0,u1,is_labeled in K_u_label_tuples:
        plotter=plot_one_K(K,u0,u1,is_labeled,plotter,n_points=num_points,rounding=rounding)
    plt.xlim(0, 1)
    plt.ylim(ylimit)
    plt.xlabel('u')
    plt.ylabel('C(u)')
    plt.title('Numerical solution C(u)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #     if u_plot.size > 1:
    #         if reached_full and not flag:
    #             flag = True
    #             print(f"First K to reach u=1: K={K}")
    #             label = f"K={K}" if K > 0 else "K=0"
    #             plt.plot(u_plot, c_plot, label=label)
    #         else:
    #             if K in labled_K_values:
    #                 label = f"K={K}" if K > 0 else "K=0"
    #                 plt.plot(u_plot, c_plot, label=label)
    #             else:
    #                 plt.plot(u_plot, c_plot)
    #     else:
    #         print(f"No valid points to plot for K={K}")


    # flag = False
    # u_eval = np.linspace(u0, u1, 1200)
    # for K in K_values:
    #     try:
    #         sol, reached_full = solve_C_for_K(K, u0=u0, u1=u1, u_eval=u_eval)
    #         if sol is None:
    #             print(f"No solution (exception) for K={K}")
    #             continue

    #         # Always plot computed portion, even if not successful
    #         u_plot = np.concatenate(([0.0], sol.t)) if sol.t.size else np.array([0.0])
    #         c_plot = np.concatenate(([0.0], sol.y[0])) if sol.t.size else np.array([0.0])
    #         # Filter out any non-finite values
    #         msk = np.isfinite(u_plot) & np.isfinite(c_plot)
    #         u_plot, c_plot = u_plot[msk], c_plot[msk]

    #         if u_plot.size > 1:
    #             if reached_full and not flag:
    #                 flag = True
    #                 print(f"First K to reach u=1: K={K}")
    #                 label = f"K={K}" if K > 0 else "K=0"
    #                 plt.plot(u_plot, c_plot, label=label)
    #             else:
    #                 if K in labled_K_values:
    #                     label = f"K={K}" if K > 0 else "K=0"
    #                     plt.plot(u_plot, c_plot, label=label)
    #                 else:
    #                     plt.plot(u_plot, c_plot)
    #         else:
    #             print(f"No valid points to plot for K={K}")

def f_lambda(x):
    # Compute lambda1 and lambda2 for beta values from 0 to 1/e
    if x <= 0 or x >= 1:
        return float('inf')
    return -x * np.log(x)

def compute_lambdas(beta):
    if beta < 0 or beta > 1/np.e:
        raise ValueError("Beta must be in the range [0, 1/e]")
    if beta == 0:
        return 0, 1
    # Small root (lambda1) in interval (0, 1/e)
    root1 = root_scalar(lambda x: f_lambda(x) - beta, bracket=(1e-10, 1/np.e), method='brentq')
    lambda1 = root1.root if root1.converged else None

    # Large root (lambda2) in interval (1/e, 1)
    root2 = root_scalar(lambda x: f_lambda(x) - beta, bracket=(1/np.e, 1-1e-10), method='brentq')
    lambda2 = root2.root if root2.converged else None
    return lambda1, lambda2

def optimal_alpha_beta(density):
    betas=[]
    alphas=[]
    start=1e-6
    end=1/np.e-2*1e-3
    # compute integer number of points from the desired step size `density`
    # ensure at least one point
    n_points = max(1, int(np.floor((end - start) / density)) + 1)
    for beta in np.linspace(start, end, n_points):
        lambda1, lambda2 = compute_lambdas(beta)
        K_low, K_high = search_optimal_K(u0=lambda1, u1=lambda2, K_start=0.01, K_end=1.0,max_iter=20)
        print(f"beta={beta}, lambda1={lambda1}, lambda2={lambda2}, K_low={K_low}, K_high={K_high}")
        alpha=K_high if K_high is not None else np.nan
        betas.append(beta)
        alphas.append(alpha)
    return np.array(betas), np.array(alphas)


if __name__ == "__main__":
    # K_u_label_tuples=[ (K,0.0,1.0,True) for K in np.linspace(0.7006050,0.7006100,11)]
    K_u_label_tuples=[ (K,0.0,1.0,True) for K in np.arange(0.0,1.1,0.1)]
    plot_batch(K_u_label_tuples=K_u_label_tuples,ylimit=(0,10),rounding=4,num_points=5000)
    # K_low, K_high = search_optimal_K(u0=0.2, u1=0.8, K_start=0.01, K_end=1.0,max_iter=20)
    # print(f"Optimal K between u0=0.2 and u1=0.8: K_low={K_low}, K_high={K_high}")
    # exit()
    # K_u_label_tuples=[ (K,0.2,0.8,True) for K in np.linspace(0.0,1.0,100)]
    # plot_batch(K_u_label_tuples=K_u_label_tuples,ylimit=(-20,20),num_points=5000)


    # K_values = np.linspace(0, 1, 10)
    # solve_and_plot(K_values,range=10)
    # search for solutions for all kinds of lambda
    # K_low, K_high = search_optimal_K(u0=1e-6, u1=1.0, K_start=0.01, K_end=1.0,max_iter=20)

    # test of solving the case u0=1e-6, u1=1.0
    # solve_and_plot(K_values=np.linspace(K_low, K_high, 10))
    # solve_and_plot(K_values=np.linspace(0.70062, 0.70063, 10))

    betas=np.linspace(1e-6, 1/np.e-2*1e-3, 10)
    alphas_1=[]
    alphas_2=[]
    for beta in betas:
        lambda1, lambda2 = compute_lambdas(beta)
        K_low, K_high = search_optimal_K(u0=lambda1, u1=lambda2, K_start=0.01, K_end=1.0,max_iter=20,n_points=1000)
        alpha_1=K_high if K_high is not None else np.nan
        alphas_1.append(alpha_1)
        sol, reached_full = solve_C_for_K(alpha_1, u0=lambda1, u1=lambda2, n_points=1000)
        C1=sol.y[0][-1] if sol is not None and sol.y.shape[1]>0 else None
        # # check if C1 satisfies the denominator condition
        # print(f"term={alpha_1-lambda2*(np.exp(C1*(1-lambda2))-1)/C1}")
        u,alpha2=compute_min_ratio(alpha=alpha_1, C=C1, lambda2=lambda2,z_max=1000.0, if_plot=False)
        u_prime,alpha2_prime=modified_compute_min_ratio(alpha=alpha_1, C=C1, lambda2=lambda2,z_max=1000.0, if_plot=False)
        alphas_2.append(alpha2)
        # print(f"beta={beta}, lambda1={lambda1}, lambda2={lambda2}, alpha_1={alpha_1}, alpha_2={alpha2}, C={C1}, u_opt={u}, u_opt_prime={u_prime}, alpha_2_prime={alpha2_prime}")
        print(f"{beta:.5f}, {lambda1:.5f}, {lambda2:.5f}, {alpha_1:.5f}, {C1:.5f}, {u:.5f}, {alpha2_prime:.5f}")
 
    '''
    betas,alphas=optimal_alpha_beta(density=0.1)
    modify_alphas=[]
    #check the solution
    for beta,alpha in zip(betas,alphas):
        lambda1, lambda2 = compute_lambdas(beta)
        sol, reached_full = solve_C_for_K(alpha, u0=lambda1, u1=lambda2, n_points=5000)
        C1=sol.y[0][-1] if sol is not None and sol.y.shape[1]>0 else None
        # modified_alpha=modify_C(lambda2,C1)
        # print(f"beta={beta}, alpha={alpha}, lambda1={lambda1}, lambda2={lambda2}, C(lambda2)={C1}, modified_alpha={modified_alpha}, expected_C={compute_limit_C(lambda2,alpha)}")
        # modify_alphas.append(modified_alpha)
    '''
        
    # plt.plot(modify_alphas,betas,label='modified curve of alpha,beta at n=inf')
    plt.plot(alphas_1,betas,label='optimal solution for the differential equation')
    plt.plot(alphas_2,betas,label='optimal solution for the min ratio')
    plt.legend()
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.title('Pareto curve of (beta, alpha)')
    plt.grid(True)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()