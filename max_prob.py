import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import math
from scipy.interpolate import interp1d
from scipy.integrate import dblquad
import warnings
from scipy.integrate import quad

def f_i(x, i, n):
    term1 = sum(x**k / ((n-k)*k) for k in range(i, n))
    term2 = (x**n / n) * (1 + sum(1 / (n-k) for k in range(i, n)))
    return term1 - term2

def g_i(x, i, n):
    # Compute g_i(x, i, n) as described
    if x == 0:
        return float('inf')  # Handle division by zero
    term = sum(((1 / x)**(n-k) - 1) / (n-k) for k in range(i, n))
    return term - 1

def find_maximum(i, n):
    # Find the maximum of f_i(x) in the interval (0, 1)
    result = root_scalar(g_i, args=(i, n), bracket=(0, 1), method='brentq')
    if result.converged:
        x_max = result.root
        y_max = f_i(x_max, i, n)
        return x_max, y_max
    else:
        return None, None

class Data:
    def __init__(self, n):
        self.n = n
        self.f_values = {}
        self.x_values = {}
        for i in range(1, n + 1):
            self.f_values[(1,i)]= f_i(1, i, n)
            self.f_values[(0,i)]= f_i(0, i, n)
            x_max, y_max = find_maximum(i, n)
            self.x_values[i]= x_max
            self.f_values[(x_max,i)]= y_max
    
    def approx_thres_function(self):
        """
        Creates an interpolation function for the threshold values x_values.
        The function maps the normalized index i/n to the corresponding x_max value.
        """
        x_vals = self.x_values
        n = self.n
        # The sample points for interpolation are i/n for i=1,...,n
        sample_points = np.array([i / n for i in range(1, n + 1)])
        # The values to interpolate are the x_max values
        threshold_values = np.array([x_vals[i] for i in range(1, n + 1)])
        # Create a linear interpolation function.
        # 'fill_value="extrapolate"' allows evaluation outside of the original range [1/n, 1].
        interp_func = interp1d(sample_points, threshold_values, kind='linear', fill_value="extrapolate")
        return interp_func
    
    def plot_approx_threshold_function(self):
        x_vals = self.x_values
        n = self.n
        # The sample points for interpolation are i/n for i=1,...,n
        sample_points = np.array([i / n for i in range(1, n + 1)])
        # The values to interpolate are the x_max values
        threshold_values = np.array([x_vals[i] for i in range(1, n + 1)])
        # Plotting the interpolation function for visualization
        plt.figure(figsize=(10, 6))
        
        # Plot the original data points
        plt.plot(sample_points, threshold_values, 'o', label='Original data (x_max values)')
        
        # Create a finer set of points for a smooth plot of the interpolation
        plot_x = np.linspace(sample_points.min(), sample_points.max(), 500)
        plot_y = self.approx_thres_function()(plot_x)
        
        # Plot the interpolated function
        plt.plot(plot_x, plot_y, '-', label='Interpolated threshold function')
        
        plt.title(f'Interpolated Threshold Function for n={self.n}')
        plt.xlabel('Normalized Index (i/n)')
        plt.ylabel('Threshold Value (x_max)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def calcu_integral(self):
        """
        Calculates the double integral:
        int_0^1 int_0^t ((1-t+t*thres(s))^n - ((1-t)*thres(t)+t*thres(s))^n) / (t*(1-t)) ds dt
        using numerical integration.
        """

        n = self.n
        thres_func = self.approx_thres_function()

        def integrand(s, t):
            # Handle potential division by zero at t=0 and t=1.
            # The limits of the integral will approach these points, but dblquad should handle it.
            # If t is very close to 0 or 1, the denominator is close to 0.
            if t == 0 or t == 1:
                return 0.0  # The contribution at a single point is zero.
            
            thres_s = thres_func(s)
            thres_t = thres_func(t)
            
            term1 = (1 - t + t * thres_s)**n
            term2 = ((1 - t) * thres_t + t * thres_s)**n
            
            numerator = term1 - term2
            denominator = t * (1 - t)
            
            return numerator / denominator

        # The outer integral is with respect to t from 0 to 1.
        # The inner integral is with respect to s from 0 to t.
        # dblquad(func, a, b, gfun, hfun) integrates func(y, x) dy dx
        # Here, y is s, and x is t.
        # So, func(s, t), x from 0 to 1, y from 0 to x (which is t).
        
        # Suppress integration warnings that might occur due to singularities
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            integral_value, error = dblquad(integrand, 0, 1, lambda t: 0, lambda t: t)

        print(f'Calculated Integral for n={self.n}: {integral_value:.4f}')
        return integral_value

    def simulate_probability_take_largest_item(self, num_trials=10000):
        count=0
        approx_threshold = self.approx_thres_function()
        for _ in range(num_trials):
            # Simulate n i.i.d. uniform random variables
            values = np.random.uniform(0, 1, self.n)
            arrival_times = np.random.uniform(0, 1, self.n)

            # Find the index of the item with the maximum value
            max_value_idx = np.argmax(values)
            max_value = values[max_value_idx]
            max_value_arrival_time = arrival_times[max_value_idx]
            
            # Find items that arrived before the item with the maximum value
            preceding_mask = arrival_times < max_value_arrival_time
            preceding_values = values[preceding_mask]
            
            # Find the maximum value among the preceding items
            if preceding_values.size > 0:
                preceding_max_value = np.max(preceding_values)
                # Find the arrival time of that preceding max value item
                preceding_max_value_idx = np.where((values == preceding_max_value) & preceding_mask)[0][0]
                preceding_max_arrival_time = arrival_times[preceding_max_value_idx]
            else:
                # If no items arrived before, set value and time to 0
                preceding_max_value = 0.0
                preceding_max_arrival_time = 0.0
            
            if max_value > approx_threshold(max_value_arrival_time) and preceding_max_value < approx_threshold(preceding_max_arrival_time):
                count += 1
        probability = count / num_trials
        return probability

    def test(self):
        # testing the validity of the algorithm
        def calculate_best_case(self):
            # consider the known distribution case where d_i=x_i_star
            total_prob = 0.0
            for i in range(1, self.n + 1):
                total_prob += self.f_values[(self.x_values[i], i)]
            print(f'Best Case Probability for n={self.n}: {total_prob:.4f}')

        def calculate_worst_case(self):
            # consider the secretary algorithm of unknown distribution where d_1,...,d_{n/e} is 1 and d_{n/e+1},...,d_n is 0
            total_prob = 0.0
            for i in range(1, int(self.n/math.e)+1):
                total_prob += self.f_values[(1, i)]
            print(f'Worst Case Probability for n={self.n}: {total_prob:.4f}')
        calculate_best_case(self)
        calculate_worst_case(self)

    def plot_f_is(self):
        # Plot the functions f_i(x) for i=1 to n
        n=self.n
        x = np.linspace(0, 1, 500)
        plt.figure(figsize=(10, 6))

        total_prob=0.0
        
        for i in range(1, n+1):
            y = [f_i(x_val, i, n) for x_val in x]
            plt.plot(x, y, label=f'f_{i}(x)')
            
            # Find and plot the maximum
            x_max, y_max = self.x_values[i], self.f_values[(self.x_values[i], i)]
            if x_max is not None:
                plt.plot(x_max, y_max, 'ro', markersize=4)  # Mark the maximum with a smaller red dot
                # plt.text(x_max, y_max, f'({x_max:.2f}, {y_max:.2f})', fontsize=8, color='red')
                total_prob += y_max
        
        print(f'Total Probability: {total_prob:.4f}') 
        plt.title(f'Functions f_i(x) for i=1 to {n}')
        plt.xlabel('x')
        plt.ylabel('f_i(x)')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_j_k(self):
        def calculate_alpha_beta(self,j,k):
            n=self.n
            alpha=1/n
            beta=1/n
            for i in range(1,j+1):
                alpha += self.f_values[(1, i)]
                beta += self.f_values[(1, i)]
            temp=0
            for i in range(j+1,k):
                alpha += self.f_values[(self.x_values[i], i)]
                temp+= self.f_values[(1, i)]
            beta += min(temp,0)
            return alpha, beta
        def sum_j_k(self,a,b):
            if b>=self.n:
                return 0
            temp=0
            for i in range(a+1,b+1):
                temp += self.f_values[(1, i)]
            return temp
        # print(calculate_alpha_beta(self, 0,self.n))
        # print(calculate_alpha_beta(self, int(self.n/math.e),int(self.n/math.e)))
        n= self.n
        alpha_values = []
        beta_values = []
        pareto_alpha_values = []
        pareto_beta_values = []
        pareto_j_values = []
        pareto_k_values = []
        for j in range(0, int(n/math.e)):
            for k in range(int(n/math.e), n+1):
                alpha, beta = calculate_alpha_beta(self, j, k)
                alpha_values.append(alpha)
                beta_values.append(beta)
                if sum_j_k(self, j, k) * sum_j_k(self, j, k+1) >= 0:
                    continue
                pareto_alpha_values.append(alpha)
                pareto_beta_values.append(beta)
                pareto_j_values.append(j)
                pareto_k_values.append(k)
        # Compute lambda1 and lambda2 for beta values from 0 to 1/e
        def f_lambda(x):
            if x <= 0 or x >= 1:
                return float('inf')
            return -x * np.log(x)

        lambda1_values = []
        lambda2_values = []
        beta_range = np.linspace(0, 1/math.e, n)

        for beta_val in beta_range:
            if beta_val == 0:
                lambda1_values.append(0)
                lambda2_values.append(1/math.e)
            else:
                # Find roots of f(x) = beta_val
                try:
                    # Small root (lambda1) in interval (0, 1/e)
                    root1 = root_scalar(lambda x: f_lambda(x) - beta_val, bracket=(1e-10, 1/math.e), method='brentq')
                    lambda1 = root1.root if root1.converged else None
                    
                    # Large root (lambda2) in interval (1/e, 1)
                    root2 = root_scalar(lambda x: f_lambda(x) - beta_val, bracket=(1/math.e, 1-1e-10), method='brentq')
                    lambda2 = root2.root if root2.converged else None
                    
                    lambda1_values.append(lambda1)
                    lambda2_values.append(lambda2)
                except:
                    lambda1_values.append(None)
                    lambda2_values.append(None)
        print(lambda1_values)
        print(lambda2_values)
        lambda_alpha=[]
        lambda_beta=[]
        for i in range(len(lambda1_values)):
            if lambda1_values[i] is not None and lambda2_values[i] is not None:
                alpha, beta = calculate_alpha_beta(self, int(lambda1_values[i]*n), int(lambda2_values[i]*n))
                lambda_alpha.append(alpha)
                lambda_beta.append(beta)
        
        def calculate_S(lambda1, lambda2):
            """
            Calculate S = integral from lambda1 to lambda2 of (integral from alpha/(1-alpha) to 1/(1-alpha) of e^(-u)/u du) dalpha
            """
            
            def inner_integrand(u):
                return np.exp(-u) / u
            
            def outer_integrand(alpha):
                if alpha >= 1:
                    return 0  # Avoid division by zero
                a = alpha / (1 - alpha)
                b = 1 / (1 - alpha)
                integral, _ = quad(inner_integrand, a, b)
                return integral
            
            # Calculate the double integral
            S, _ = quad(outer_integrand, lambda1, lambda2)
            return S

        def calculate_alpha_beta_from_lambda(lambda1, lambda2):
            alpha1=f_lambda(lambda1)
            alpha2=calculate_S(lambda1, lambda2)
            beta2=f_lambda(lambda2)
            return alpha1+alpha2, min(alpha1,beta2)
        
        lambda_analytical_alpha=[]
        lambda_analytical_beta=[]
        # Plot approximate analytical values
        for i in range(len(lambda1_values)):
            if lambda1_values[i] is not None and lambda2_values[i] is not None:
                alpha,beta= calculate_alpha_beta_from_lambda(lambda1_values[i], lambda2_values[i])
                lambda_analytical_alpha.append(alpha)
                lambda_analytical_beta.append(beta)


        plt.figure(figsize=(10, 6))
        # plt.plot(alpha_values, beta_values, 'bo-', label='Alpha vs Beta')
        plt.plot(pareto_alpha_values, pareto_beta_values, 'r--', label='Pareto Alpha-Beta')  # Add Pareto points in red
        plt.plot(lambda_alpha, lambda_beta, 'g--', label='Lambda1 Alpha') # Add lambda1 points in green dashed line
        plt.plot(lambda_analytical_alpha, lambda_analytical_beta, 'm-.', label='Lambda Analytical Alpha') # Add analytical lambda points in magenta dash-dot line
        plt.title(f'Alpha vs Beta for n={n}')
        plt.xlabel('Alpha')
        plt.ylabel('Beta')
        plt.xlim(0, 1)  # Fix x-axis range to [0, 1]
        plt.ylim(0, 1)  # Fix y-axis range to [-1, 1]
        plt.legend()
        plt.grid()
        plt.show()

        # Plot (j/n, k/n) for Pareto points
        plt.figure(figsize=(10, 6))
        pareto_j_normalized = [j/n for j in pareto_j_values]
        pareto_k_normalized = [k/n for k in pareto_k_values]
        plt.plot(pareto_j_normalized, pareto_k_normalized, 'ro-', label='Pareto (j/n, k/n)')
        
        # Plot lambda1 and lambda2 values
        valid_lambda1 = [val for val in lambda1_values if val is not None]
        valid_lambda2 = [val for val in lambda2_values if val is not None]
        if valid_lambda1 and valid_lambda2:
            plt.plot(valid_lambda1, valid_lambda2, 'b^-', label='Lambda1 vs Lambda2')
        
        plt.title(f'Pareto Points (j/n, k/n) for n={n}')
        plt.xlabel('j/n')
        plt.ylabel('k/n')
        plt.legend()
        plt.grid()
        plt.show()
        
# Example usage
n=100
data=Data(n)
# output=data.x_values
# import pandas as pd
# df = pd.DataFrame({'Value': [i/n for i in range(1, n+1)],
#                    'Output': [output[i] for i in range(1, n+1)]})
# df.to_clipboard(index=False, sep='\t')

# print(data.simulate_probability_take_largest_item(10000))
# data.calcu_integral()
# data.plot_approx_threshold_function()
# data.test()
# data.plot_f_is()
data.plot_j_k()
# data=Data(100)

