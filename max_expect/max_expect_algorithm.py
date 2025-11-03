import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.integrate import quad
import os
import re
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
        return 0.0, 1.0

    # Small root (lambda1) in (0, 1/e)
    root1 = root_scalar(lambda x: f_lambda(x) - beta, bracket=(1e-12, 1 / np.e), method="brentq")
    lambda1 = root1.root if root1.converged else None

    # Large root (lambda2) in interval (1/e, 1)
    root2 = root_scalar(
        lambda x: f_lambda(x) - beta, bracket=(1 / np.e, 1 - 1e-10), method="brentq"
    )
    lambda2 = root2.root if root2.converged else None
    return lambda1, lambda2

def solve_init_theta_lambda2(lambda2,alpha):
    """Solve the boundary condition at lambda2:

    Solve for theta in the equation
        lambda2 * \int_{lambda2}^{1} theta^t / t dt = alpha * theta
    using a robust bisection on theta in (0, 1].

    Returns (theta, residual) where residual is equation(theta).
    """
    def integrand(t, theta_val):
        return np.exp( t * np.log(theta_val)) / t

    def equation(theta):
        integral_val, _ = quad(integrand, lambda2, 1, args=(theta,))
        return lambda2 * integral_val - alpha * theta

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
    """Compute a feasible threshold function theta(z) on [lambda1, lambda2].

    Parameters
    - beta, alpha: problem parameters as in the paper.
    - num_steps: number of uniform subdivisions between lambda1 and lambda2.
    - if_plot: if True, plot the resulting step-function.
    """
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

if __name__ == "__main__":
    # Numerical driver: read alpha and beta values from alpha_betas.txt and
    # compute feasible threshold functions for each pair.
    file_path = os.path.join(os.path.dirname(__file__), "alpha_betas.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    def _extract(name: str) -> np.ndarray:
        m = re.search(rf"{name}\s*=\s*\[(.*?)\]", content, re.DOTALL)
        if not m:
            raise ValueError(f"{name} array not found in alpha_betas.txt")
        nums = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", m.group(1))
        return np.array([float(x) for x in nums], dtype=float)

    betas = _extract("betas")
    alphas = _extract("alphas")

    for beta, alpha in zip(betas, alphas):
        # # test: plot theresholds
        # threshold_vals=solve_de_recursively(beta,alpha,num_steps=100,if_plot=True)
        threshold_vals=solve_de_recursively(beta,alpha,num_steps=300)
        out_path = os.path.join(os.path.dirname(__file__), "valid_threshold_functions.txt")
        lambda1, lambda2 = compute_lambdas(beta)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(f"beta={beta}, lambda1={lambda1}, lambda2={lambda2}, alpha={alpha}, threshold_vals={threshold_vals}\n")
        if threshold_vals[0]>1.0:
            print(f"beta={beta}, lambda1={lambda1}, lambda2={lambda2}, alpha={alpha}, threshold at lambda1={threshold_vals[0]}") 
        else:
            print("Warning: Threshold at lambda1 does not exceed 1.0. Skip this beta.")
            alphas.append(0)