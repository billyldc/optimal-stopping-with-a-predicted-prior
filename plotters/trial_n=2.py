import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings
from scipy.optimize import root_scalar
from opt_analytical_threshold import optimal_alpha_beta
warnings.filterwarnings('ignore')

def solve_threshold_equation(alpha, n_points=1000,n=2,method='Radau',start=0,end=1):
    """
    Solve the ODE for the threshold function
    θ'(u) = -(1 - u + u θ(u))^{n-1}/(u(θ(u)^{n-1}-(1-u+u θ(u))^{n-1})/(1-θ(u)) + α (n-1) θ(u)^{n-2}
    θ(0) = 1, θ(1) = 0
    """
    def ode(u, theta):
        ## We actually calculate (θ(u)^{n-1}-(1-u+u θ(u))^{n-1})/(1-θ(u)) by (u-1)*sum_{k=0}^{n-2} θ(u)^k (1-u+u θ(u))^{n-2-k} to avoid numerical instability when θ(u) is close to 1
        series_sum = 0
        for k in range(n-1):
            series_sum += theta[0]**k * (1 - u + u * theta[0])**(n - 2 - k)
        denominator = u * (series_sum)*(u-1) + alpha * (n - 1) * theta[0] ** (n - 2)
        # Avoid division by zero by adding a small amount
        denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
        numerator = -(1 - u + u * theta[0])**(n-1)
        return [numerator / denominator]
    
    # Use solve_bvp to solve the boundary value problem
    try:
        sol = solve_ivp(
                fun=ode,
                t_span=(start, end),
                y0=[1],
                t_eval= np.linspace(start, end, n_points),
                method=method,
                rtol=1e-6,
                atol=1e-6,
                # events=events,
            )
        return sol.t, sol.y[0], len(sol.y[0])==n_points
    except Exception as e:
        print(f"α={alpha:.2f}: Solution failed - {e}")
        return None, None,False

# def additional_term_l_0()

def plot_solutions(alpha_n_pairs=[(a,2) for a in np.linspace(0.5, 1.0, 15)], start=0.0, end=1.0):
    """
    Plot the solutions for different alpha values
    """
    plt.figure(figsize=(12, 8))

    for alpha, n in alpha_n_pairs:
        print(f"Solving α = {alpha:.2f} with n={n}...")
        
        u, theta,_ = solve_threshold_equation(alpha, n=n,start=start,end=end)
        
        if u is not None:
            # map alpha to a smooth colormap (larger alpha -> different color along the viridis gradient)
            alphas = [p[0] for p in alpha_n_pairs]
            min_a, max_a = min(alphas), max(alphas)
            norm = (alpha - min_a) / (max_a - min_a) if max_a > min_a else 0.5
            norm = np.clip(norm, 0.0, 1.0)
            cmap = plt.cm.viridis  # try 'plasma', 'inferno', 'magma' for other gradients
            color = cmap(norm)
            plt.plot(u, theta, color=color, label=f'n={n}, α = {alpha:.6f}')

    plt.xlabel('u', fontsize=12)
    plt.ylabel('θ(u)', fontsize=12)
    plt.title(f'Solutions for threshold function θ(u)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.show()

def search_solutions(alpha_range=(0.1,1.0), n_range=[2], n_steps=10,start=0.0,end=1.0):
    """
    Binary Search for solutions over the range
    """
    searched_alpha_values=[]
    for n in n_range:
        search_start=alpha_range[0]
        search_end=alpha_range[1]
        for _ in range(n_steps):
            search_mid = (search_start + search_end) / 2
            print(f"Searching α in [{search_start:.6f}, {search_end:.6f}], mid={search_mid:.6f}, n={n}")
            u, theta,is_success = solve_threshold_equation(search_mid, n=n,start=start,end=end)
            if min(theta)>0 and is_success:
                search_end=search_mid
            else:
                search_start=search_mid
        print(f"Search complete. n={n}, Approximate α = {search_mid:.6f}")
        searched_alpha_values.append((search_mid,n))
    return searched_alpha_values

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

if __name__ == "__main__":
    # example usage
    plot_alpha_n_pairs=[(a,4) for a in np.linspace(0.5, 0.8, 20)]
    plot_solutions(plot_alpha_n_pairs,start=0.1,end=0.9)
    # lambda1, lambda2 = compute_lambdas(0.3)
    # plot_solutions(plot_alpha_n_pairs,start=lambda1,end=lambda2)
    # solved_alpha_n_pairs=search_solutions(n_range=range(2,30), n_steps=20)
    # plot_solutions(solved_alpha_n_pairs)
    betas=np.linspace(1e-6, 1/np.e-1e-2, 100)
    alphas_2=[]
    alphas_n=[]
    for beta in betas:
        lambda1, lambda2 = compute_lambdas(beta)
        print(f"beta={beta}, lambda1={lambda1}, lambda2={lambda2}")
        list_sol=search_solutions(alpha_range=(0.1, 1.0), n_range=[3], n_steps=15,start=lambda1,end=lambda2)
        alphas_2.append(list_sol[0][0])
    print("Final alpha solutions:", alphas_2)
    betas_n,alphas_n=optimal_alpha_beta(density=1e-2)
    plt.plot(alphas_2,betas,label='curve of alpha,beta at n=2')
    plt.plot(alphas_n,betas_n,label='curve of alpha,beta at n=infty')
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.legend()
    # plt.title('Pareto curve of (beta, alpha)')
    plt.grid(True)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()
