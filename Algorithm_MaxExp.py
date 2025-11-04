import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.integrate import quad
import os
from helper import read_data


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


def solve_de_recursively(beta,alpha,num_steps=100,if_plot=False):
    """Compute a feasible threshold function theta(z) on [lambda1, lambda2].

    Parameters
    - beta, alpha: problem parameters as in the paper.
    - num_steps: number of uniform subdivisions between lambda1 and lambda2.
    - if_plot: if True, plot the resulting step-function.
    """
    lambda1, lambda2 = compute_lambdas(beta)
    m=num_steps
    dz = (lambda2 - lambda1) / m
    threshold_vals=[1.0]*(m+1)
    for i in range(m,0,-1):
        z_ip1= lambda1 + i * dz
        def inner_integrand1(t, theta_val):
            return np.exp( t * np.log(theta_val)) / t
        def inner_integrand2(t, theta_val,z_lower):
            return np.exp( t * np.log(theta_val))*(t-z_lower) / t
        def equation(theta):
            # z_{i+1} int_{z_{i+1}}^1 frac{ \theta_i^t}{t} \,\dd t 
            term1= z_ip1 * quad(inner_integrand1, z_ip1, 1, args=(theta,))[0]
            """
                sum_{j=i+1}^m left[ int_{z_{j}}^{z_{j+1}} frac{t- z_j}{t} dt  {threshold_j^*}^t 
            + (z_{j+1}-z_j) int_{z_{j+1}}^1 frac{{threshold^*_j}^t}{t}  dt]
            """
            term2= sum(quad(inner_integrand2, (lambda1+(j-1)*dz), (lambda1+j*dz), args=(threshold_vals[j],(lambda1+(j-1)*dz),))[0]
                       +quad(inner_integrand1, (lambda1+j*dz), 1, args=(threshold_vals[j],))[0]*dz 
                       for j in range(i+1,m+1))
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

    threshold_vals[0] = threshold_vals[1]  # supplimentary the value at z = lambda1

    # plot the thresholds as a piecewise-constant (step) function:
    if if_plot:
        z_edges = np.linspace(lambda1, lambda2, m + 1)
        # For 'post': y[i] applies on [x[i], x[i+1]); repeat last value to keep flat at the end
        y_steps = np.empty(m + 1)
        y_steps[:-1] = threshold_vals[1:]
        y_steps[-1] = threshold_vals[-1]

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
    """
    This code implements the numerical verification step in Appendix B.
    Verify that for each (alpha, beta) pair from alpha_betas.txt, we can compute a feasible threshold function
    via solving the differential equation recursively.
    First, extract alpha and beta values from Algorithm_MaxExp.txt
    Then, for each pair, compute the threshold function and save to valid_threshold_functions.txt
    """

    #Read alpha and beta values from alpha_betas.txt
    alphas, betas = read_data(os.path.join(os.path.dirname(__file__), "Algorithm_MaxExp.txt"))

    out_path = os.path.join(os.path.dirname(__file__), "valid_threshold_functions.txt")

    for beta, alpha in zip(betas, alphas):
        ## test: plot theresholds
        # threshold_vals=solve_de_recursively(beta,alpha,num_steps=100)
        ## plot if needed
        # threshold_vals=solve_de_recursively(beta,alpha,num_steps=300,if_plot=True)

        threshold_vals=solve_de_recursively(beta,alpha,num_steps=300,if_plot=False)
        lambda1, lambda2 = compute_lambdas(beta)
        # print(threshold_vals)
        # exit(0)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(f"beta={beta}, lambda1={lambda1}, lambda2={lambda2}, alpha={alpha}, threshold_vals={threshold_vals}\n")
        if threshold_vals[0]>1.0:
            print(f"beta={beta}, lambda1={lambda1}, lambda2={lambda2}, alpha={alpha}, threshold at lambda1={threshold_vals[0]}") 
        else:
            print("Warning: Threshold at lambda1 does not exceed 1.0. Skip this beta.")