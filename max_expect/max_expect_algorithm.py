import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.integrate import quad
import os
import matplotlib.pyplot as plt

# auxiliary functions
def f_lambda(x):
    # Compute lambda1 and lambda2 for beta values from 0 to 1/e
    if x <= 0 or x >= 1:
        return float("inf")
    return -x * np.log(x)


def compute_lambdas(beta):
    if beta < 0 or beta > 1 / np.e:
        raise ValueError("Beta must be in the range [0, 1/e]")
    if beta == 0:
        return 0, 1
    # Small root (lambda1) in interval (0, 1/e)
    root1 = root_scalar(
        lambda x: f_lambda(x) - beta, bracket=(1e-10, 1 / np.e), method="brentq"
    )
    lambda1 = root1.root if root1.converged else None

    # Large root (lambda2) in interval (1/e, 1)
    root2 = root_scalar(
        lambda x: f_lambda(x) - beta, bracket=(1 / np.e, 1 - 1e-10), method="brentq"
    )
    lambda2 = root2.root if root2.converged else None
    return lambda1, lambda2

def solve_init_theta_lambda2(lambda2,alpha):
    """
    Solve the equation lambda2 *\int_{lambda2}^{1} theta(lambda2)^t / t dt = alpha *theta(lambda2)
    """
    def integrand(t, theta_val):
        return np.exp( t * np.log(theta_val)) / t

    def equation(theta):
        I, _ = quad(integrand, lambda2, 1, args=(theta,))
        return lambda2 * I - alpha * theta

    root_low=0.0
    root_high=1.0
    for _it in range(50):
        mid=0.5 * (root_low + root_high)
        if equation(mid)<0:
            root_high= 0.5 * (root_low + root_high)
        else:
            root_low= 0.5 * (root_low + root_high)
    return root_low, equation(root_low)

def solve_de_recursively(beta,alpha,num_steps=100,if_plot=False):
    lambda1, lambda2 = compute_lambdas(beta)
    theta_lambda2, _ = solve_init_theta_lambda2(lambda2, alpha)
    m=num_steps
    dz = (lambda2 - lambda1) / m
    threshold_vals=[1.0]*(m+1)
    for i in range(m,0,-1):
        z_ip1= lambda1 + i * dz
        def integrand(t, theta_val):
            return np.exp( t * np.log(theta_val)) / t
        def equation(theta):
            term1= z_ip1 * quad(integrand, z_ip1, 1, args=(theta,))[0]
            term2= dz * sum(quad(integrand, lambda1 + j * dz, 1, args=(threshold_vals[j-1],))[0] for j in range(i+1,m+1))
            return term1 + term2 - alpha * theta
        root_low=0.0
        root_high=2.0
        for _it in range(100):
            mid=0.5 * (root_low + root_high)
            if equation(mid)<0:
                root_high= 0.5 * (root_low + root_high)
            else:
                root_low= 0.5 * (root_low + root_high)
        threshold_vals[i]= root_low
    # plot the thresholds as a piecewise-constant (step) function:
    # theta(z) = threshold_vals[i] for z in [lambda1 + (i-1)*dz, lambda1 + i*dz]
    threshold_vals[0] = threshold_vals[1]  # define value at z = lambda1

    z_edges = np.linspace(lambda1, lambda2, m + 1)
    # For 'post': y[i] applies on [x[i], x[i+1]); repeat last value to keep flat at the end
    y_steps = np.empty(m + 1)
    y_steps[:-1] = threshold_vals[1:]
    y_steps[-1] = threshold_vals[-1]
    
    if if_plot:
        plt.figure()
        plt.step(z_edges, y_steps, where='post')
        plt.xlabel('z')
        plt.ylabel('theta(z)')
        plt.title(f'Threshold function theta(z) for beta={beta}, alpha={alpha}')
        plt.xlim(lambda1, lambda2)
        plt.grid(True)
        plt.show()
    return threshold_vals


def dtheta_dz(z, y, alpha):
    # denominator of theta' ODE
    # z * (1 - theta^{z-1}) / ln theta - \alpha
    # Handle small theta and near-singular denominators
    theta=y[0]
    theta_small = 1e-10
    D_small = 1e-10
    if abs(theta-1)< theta_small:
        # For theta->1, (1 - theta^{z-1})/ln theta ~ (1-z)
        denom_approx =  z * (1.0 - z) - alpha
        denom_approx = np.sign(denom_approx) * max(abs(denom_approx), D_small)
        return [1.0/denom_approx]
    denom =  z * (1.0 - np.exp((z - 1.0) * np.log(theta))) / np.log(theta) - alpha
    numerator = np.exp(z * np.log(theta))
    if abs(denom) < D_small:
        denom = np.sign(denom) * D_small
    return [numerator / denom]

def solve_theta_for_alpha(alpha, z0=0.0, z1=1.0, method='Radau', rtol=1e-10, atol=1e-10, n_points=200):
    """
    Solve the ODE for given alpha and return (sol, reached_full_z)
    reached_full_z is True when the solver produced values at all points in z_eval.
    On exception, returns (None, False).
    """
    z_eval = np.linspace(z0, z1, n_points)
    try:
        sol = solve_ivp(
            lambda z, y: dtheta_dz(z, y, alpha),
            (z0, z1),
            [1],
            t_eval=z_eval,
            method=method,
            rtol=rtol,
            atol=atol,
        )
        reached_full = sol.t.size == z_eval.size
        reached_full = reached_full and max(sol.y[0]) <= 1.0 and min(sol.y[0]) >= 0.0
        decreasing_check = np.all(np.diff(sol.y[0]) <= 1e-12)
        reached_full = reached_full and decreasing_check
        # print(f"Alpha={alpha}: reached_full={reached_full}, decreasing_check={decreasing_check}, sol.t.size={sol.t.size}, expected size={z_eval.size}, solution={sol.y[0]}")
        return sol, reached_full
    except Exception:
        return None, False

def search_optimal_alpha(z0=0.0, z1=1.0, alpha_start=0.01, alpha_end=1.0, max_iter=10, n_points=5000):
    """
    Binary search for the boundary alpha:
    find largest alpha_low such that solver does NOT reach z1 (sol.t.size < len(z_eval))
    and smallest alpha_high such that solver DOES reach z1 (sol.t.size == len(z_eval)).
    Returns (alpha_low, alpha_high) or (None, None) if no valid bracket found.
    """

    def reached_full_z(alpha):
        _, reached_full = solve_theta_for_alpha(alpha, z0=z0, z1=z1, n_points=n_points)
        return reached_full
    # # alpha_start= max(alpha_start, z0*(1.0 - z0)+ 1e-12)  # ensure alpha_start > z0(1-z0), otherwise theta' denominator is larger than 0 at z0

    # # use grid search first to find a valid bracket
    # n_grid = 20
    # alphas_grid = np.linspace(alpha_start, alpha_end, n_grid)
    # reached_alphas=[]
    # for alpha in alphas_grid:
    #     sol,reached_full=solve_theta_for_alpha(alpha, z0=z0, z1=z1, n_points=n_points)
    #     if reached_full:
    #         reached_alphas.append(alpha)

    # print("Reached alphas in grid search:", reached_alphas)
    # if len(reached_alphas)==0:
    #     print("No reaching alpha in the grid search; increase alpha_end.")
    #     return (None, None)
    # alpha_start = reached_alphas[-1]
    # alpha_end=alpha_start+ (alpha_end - alpha_start)/n_grid

    # ensure we have a valid bracket
    # intuitively, for large alpha, the solver should reach z1, while for small alpha, it should not
    # thus, we should have start_reached=False, end_reached=True
    start_reached = reached_full_z(alpha_start)
    end_reached = reached_full_z(alpha_end)

    if start_reached:
        print("modified alpha_start does reach z1; no failing alpha in given range.")
        return (None, None)
    if not end_reached:
        print("alpha_end does not reach z1; no reaching alpha in given range.")
        return (None, None)

    alpha_low, alpha_high = float(alpha_start), float(alpha_end)
    tol = 1e-10
    it = 0
    while (alpha_high - alpha_low) > tol and it < max_iter:
        alpha_mid = 0.5 * (alpha_low + alpha_high)
        # if alpha_mid reaches z1, this means alpha_mid is too high
        if reached_full_z(alpha_mid):
            alpha_high = alpha_mid
        else:
            alpha_low = alpha_mid
        it += 1

    # print(f"Search done in {it} iters. Largest failing alpha ~ {alpha_low}, smallest reaching alpha ~ {alpha_high}")
    return (alpha_low, alpha_high)

def plot_one_alpha(alpha_value, z0=0.0, z1=1.0, is_labeled=True, plotter=None, n_points=100, rounding=10):
    ## plot for one alpha_value and z0,z1
    ## label the curve if required
    try:
        sol, reached_full = solve_theta_for_alpha(alpha_value, z0=z0, z1=z1, n_points=n_points)
        if sol is None:
            print(f"No solution (exception) for alpha={alpha_value}")

        # Always plot computed portion, even if not successful
        z_plot = np.concatenate(([z0], sol.t)) if sol.t.size else np.array([z0])
        theta_plot = np.concatenate(([1.0], sol.y[0])) if sol.t.size else np.array([0.0])
        # # Filter out any non-finite values
        # msk = np.isfinite(z_plot) & np.isfinite(theta_plot)
        # z_plot, theta_plot = z_plot[msk], theta_plot[msk]
        if is_labeled:
            label = f"z0={z0}, z1={z1}, alpha={alpha_value:.{rounding}f}"
            plt.plot(z_plot, theta_plot, label=label)
        else:
            plt.plot(z_plot, theta_plot)
    except Exception as e:
        print(f"Error for alpha={alpha_value}: {e}")
    return plotter

def plot_batch(alpha_z_label_tuples, ylimit=(0,20), num_points=100, rounding=10):
    ## plot for a batch of (alpha,z0,z1) tuples
    plotter = plt.figure(figsize=(10, 6))
    for alpha, z0, z1, is_labeled in alpha_z_label_tuples:
        plotter = plot_one_alpha(alpha, z0, z1, is_labeled, plotter, n_points=num_points, rounding=rounding)
    plt.xlim(0, 1)
    plt.ylim(ylimit)
    plt.xlabel('z')
    plt.ylabel('theta(z)')
    plt.title('Numerical solution theta(z)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def optimal_alpha_beta(density):
    betas = []
    alphas = []
    start = 1e-6
    end = 1/np.e-2*1e-3
    # compute integer number of points from the desired step size `density`
    # ensure at least one point
    n_points = max(1, int(np.floor((end - start) / density)) + 1)
    for beta in np.linspace(start, end, n_points):
        lambda1, lambda2 = compute_lambdas(beta)
        alpha_low, alpha_high = search_optimal_alpha(z0=lambda1, z1=lambda2, alpha_start=0.01, alpha_end=1.0, max_iter=20)
        print(f"beta={beta}, lambda1={lambda1}, lambda2={lambda2}, alpha_low={alpha_low}, alpha_high={alpha_high}")
        alpha = alpha_high if alpha_high is not None else np.nan
        betas.append(beta)
        alphas.append(alpha)
    return np.array(betas), np.array(alphas)


if __name__ == "__main__":
    # Numerical solution mentioned in Appendix B
    # Compute a feasible threshold function for a range of beta values
    #  betas = []
    # alphas = []
    start = 1e-6
    end = 1/np.e-2*1e-3
    density=0.005
    n_points = max(1, int(np.floor((end - start) / density)) + 1)
    betas=np.linspace(start, end, n_points)
    alphas=[]
    for beta in betas:
        lambda1, lambda2 = compute_lambdas(beta)
        alpha_low, alpha_high = search_optimal_alpha(z0=lambda1, z1=lambda2, alpha_start=0.001, alpha_end=1.0, max_iter=20, n_points=5000)
        theta_lambda2,ineq_val= solve_init_theta_lambda2(lambda2,alpha_high)
        error=0.001
        alphas.append(alpha_high-error)
        continue
        threshold_vals=solve_de_recursively(beta,alpha_high-error,num_steps=300)
        out_path = os.path.join(os.path.dirname(__file__), "valid_threshold_functions.txt")
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(f"beta={beta}, lambda1={lambda1}, lambda2={lambda2}, alpha={alpha}, threshold_vals={threshold_vals}\n")
        if threshold_vals[0]>1.0:
            alphas.append(alpha)
            print(f"beta={beta}, lambda1={lambda1}, lambda2={lambda2}, alpha={alpha}, threshold at lambda1={threshold_vals[0]}") 
        else:
            print("Warning: Threshold at lambda1 does not exceed 1.0. Skip this beta.")
            alphas.append(0)
        continue

        def search_stable(lambda2_val):
            alpha_low, alpha_high = search_optimal_alpha(z0=lambda1, z1=lambda2_val, alpha_start=0.01, alpha_end=1.0, max_iter=20, n_points=5000)
            sol, reached_full = solve_theta_for_alpha(alpha_high, z0=lambda1, z1=lambda2_val, n_points=5000)
            return sol.y[0][-1], alpha_high, alpha_low
        modified_lambda2=lambda2
        modified_alpha=0.0
        while True:
            threshold_lambda2_sol, alpha_high,alpha_low = search_stable(modified_lambda2)
            if threshold_lambda2_sol<1e-3:
                modified_lambda2-=1e-2
            else:
                modified_alpha=alpha_low
                break
        plot_one_alpha(modified_alpha, z0=lambda1, z1=modified_lambda2, is_labeled=True, n_points=5000)
        print(f"beta={beta}, lambda1={lambda1}, modified_lambda2={modified_lambda2}, modified_alpha={modified_alpha}")
        plt.show()
        exit()
    out_path = os.path.join(os.path.dirname(__file__), "alpha_betas.txt")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(f"betas={betas}\n")
        f.write(f"alphas={alphas}\n")
    print(alphas,betas)
    plt.plot(betas, alphas, label='optimal solution for the differential equation')
    plt.xlabel("beta")
    plt.ylabel("alpha")
    plt.title("Pareto curve of (beta, alpha)")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


    betas = np.linspace(1e-6, 1/np.e-2*1e-3, 10)
    alphas_1 = []
    alphas_2 = []
    for beta in betas:
        lambda1, lambda2 = compute_lambdas(beta)
        alpha_low, alpha_high = search_optimal_alpha(z0=lambda1, z1=lambda2, alpha_start=0.01, alpha_end=1.0, max_iter=20, n_points=5000)
        alpha_1 = alpha_high if alpha_high is not None else np.nan
        alphas_1.append(alpha_1)
        bound_difference = boundary_difference_theta(beta, alpha=alpha_high)
        threshold_lambda2_sol, _ = solve_theta_for_alpha(alpha_1, z0=lambda1, z1=lambda2, n_points=5000)
        boundary_2_value = boundary_2(lambda2, threshold_lambda2_sol.y[0][-1],alpha_1)
        verifier_alpha_3=direct_verifier(beta, alpha_1)
        print(f"beta={beta}, lambda1={lambda1}, lambda2={lambda2}, alpha_high={alpha_high}, boundary_difference={bound_difference}, boundary_2_value={boundary_2_value}, verifier_alpha_3={verifier_alpha_3}")

    # plt.plot(modify_alphas,betas,label='modified curve of alpha,beta at n=inf')
    plt.plot(alphas_1, betas, label='optimal solution for the differential equation')
    plt.legend()
    plt.xlabel("alpha")
    plt.ylabel("beta")
    plt.title("Pareto curve of (beta, alpha)")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

