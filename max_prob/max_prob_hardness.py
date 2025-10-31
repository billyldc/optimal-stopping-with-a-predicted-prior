import numpy as np
from itertools import product
from gurobipy import *
from json import loads
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up output directory
workdir = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(workdir, "output")

class Distributions:
    """
    Represents a family of tail-truncated probability distributions.

    This class partitions a base distribution into multiple truncated versions,
    including a designated prior, and computes normalization constants and
    probability mass functions accordingly.

    Attributes:
        v (list of float): Support values of the base distribution.
        p (list of float): Probability masses corresponding to `v`.
        indices (list of int): Truncation indices for each distribution.
        cum_p_partition (list of float): Cumulative probabilities for each partition.
        v_partition (list of int): Maps each support value to its partition index.
        K (int): Total number of distributions.
        k_star (int): Index of the prior distribution in the sorted list of `indices`.
    """

    def __init__(self, probs, values = None, indices = None, prior_index = None):
        """
        Initializes the Distributions object with truncated probability distributions.
    
        Args:
            probs (list of float): Nonnegative probability masses corresponding to `values`.
            values (list of float): Strictly increasing nonnegative values representing the support. 
                                    If None, values is set to np.arange(1, len(probs)+1, 1).
            indices (list of int): Truncation indices for each distribution. If None, indices is set to list(range(len(values))).
            prior_index (int): Truncation index for the prior distribution. If None, prior_index is set to 0.
    
        Raises:
            ValueError: If any input fails validation checks.
        """        
        if not all(p >= 0 for p in probs):
            raise ValueError("All probs must be nonnegative.")

        if values is not None and not all(v >= 0 for v in values):
            raise ValueError("All values must be nonnegative.")
        if values is not None and not all(values[i] < values[i + 1] for i in range(len(values) - 1)):
            raise ValueError("Values must be strictly increasing.")
        if values is not None and len(probs) != len(values):
            raise ValueError("Values must have the same length as probs.")
        
        if indices is not None and not all(isinstance(index, int) and 0 <= index < len(probs) for index in indices):
            raise ValueError("Indices must be nonnegative integers between 0 and len(probs) - 1.")
        
        if prior_index is not None and not (isinstance(prior_index, int) and 0 <= prior_index < len(probs)):
            raise ValueError("Prior index must be an integer between 0 and len(probs) - 1.")

        sum_probs = sum(probs)
        probs = [_/sum_probs for _ in probs]
        self.p = probs
        if (values is None):
            self.v = np.arange(1, len(probs)+1, 1)
        else:
            self.v = values
        if (indices is None):
            self.indices = sorted(list(range(len(probs))), reverse=True)
        else:
            self.indices = sorted(set(indices+[prior_index]), reverse=True)
        self.K = len(self.indices)
        if (prior_index is None):
            self.k_star = 0
        else:
            self.k_star = self.indices.index(prior_index)
        self.v_partition = []
        p_partition = [0] * self.K
        l = 0
        for i in range(len(self.v)-1, -1, -1):
            if (l+1 <= len(self.indices)-1 and self.indices[l+1] == i):
                l += 1
            self.v_partition.insert(0,l)
            p_partition[l] += self.p[i]
        self.cum_p_partition = np.cumsum(p_partition[::-1])[::-1].tolist() + [0.0]

    def summarize(self, sigfig = 5):
        """
        Prints a summary of the distributions, including support and normalized PMFs.
    
        Args:
            sigfig (int): Number of significant figures to display.
        """
        print("Summary of distributions (*prior):")
        print(f"Support:\n    {[float(f'{v:.{sigfig}g}') for v in self.v]}")
        print("PMF:")
        for k in range(self.K):
            c_ = self.c(k)
            print(f"{'*' if k == self.k_star else ' '}{k}:", end=" ")
            print(
                [float(f'{p*c_:.{sigfig}g}')*(i <= self.indices[k])
                 for i, p in enumerate(self.p)]
            )
        print('\n')
            
    def I(self, x):
        """
        Returns the partition index for a given value or index.
    
        Args:
            x (int or list of int): Index or list of indices to evaluate.
    
        Returns:
            int: Partition index corresponding to `x`.
        """
        if (x is None):
            return self.K - 1
        if (type(x) == int):
            return self.v_partition[x]
        return self.v_partition[max(x)]

    def c(self, k):
        """
        Returns the normalization constant for distribution `k`.
    
        Args:
            k (int): Distribution index.
    
        Returns:
            float: Inverse of cumulative probability for partition `k`.
        """
        return 1/self.cum_p_partition[k]

    def P(self, t, l, mode):
        """
        Computes the probability of a sequence being in a given state under distribution `0`.
    
        Args:
            t (int): Sequence length.
            l (int): Partition index.
            mode (str): Either ">=" or "==", specifying the condition.
    
        Returns:
            float: Computed probability based on the mode.

            - If mode is ">=":
                Returns the probability that the largest of `t` elements 
                fall within partition `l` or higher.

            - If mode is "==":
                Returns the probability that the largest of `t` elements 
                fall exactly within partition `l`.

            - If `t == 0`:
                - For ">=": Always returns 1 (empty sequence is trivially valid).
                - For "==": Returns 1 if `l` is the last partition (`K - 1`), else 0.
    
        Raises:
            ValueError: If `mode` is not recognized.
        """
        if (mode == ">="):
            if (t == 0):
                return 1
            return self.cum_p_partition[l]**t
        if (mode == "=="):
            if (t == 0):
                return (1 if l == self.K-1 else 0)
            return self.cum_p_partition[l]**t - self.cum_p_partition[l+1]**t
        

class LP_maxprob:
    """
    Constructs and solves a linear programming model to optimize consistency and robustness 
    in a sequence of truncated probability distributions using Gurobi.

    This model evaluates the trade-off between two metrics—alpha (consistency) and beta (robustness)— 
    based on a given family of truncated distributions and sequence length.

    Attributes:
        n (int): Length of the arrival sequence.
        F (Distributions): Instance of the Distributions class representing the truncated distributions.
        model (gurobipy.Model): Gurobi optimization model.
        alpha (float or None): Consistency parameter (or `None` if optimized).
        beta (float or None): Robustness parameter (or `None` if optimized).
        S (gurobipy.VarDict): Decision variables representing survival probabilities.
    """
    
    def __init__(self, n, F, alpha, beta, outputflag = 0, monotone = False):
        """
        Initializes the LP model and builds the optimization problem.
    
        Args:
            n (int): Length of the arrival sequence.
            F (Distributions): Instance of the Distributions class.
            alpha (float or None): Consistency requirement. Must be None if beta is provided.
            beta (float or None): Robustness requirement. Must be None if alpha is provided.
            outputflag (int): 0 to suppress solver output, 1 to enable it.
    
        Raises:
            ValueError: If `n` is not a positive integer.
        """
        if not (isinstance(n, int) and n > 0):
            raise ValueError("n must be a positive integer.")
        if not ((alpha is None and isinstance(beta, (float,int)) and 0 <= beta <= 1) or 
                (isinstance(alpha, (float,int)) and 0 <= alpha <= 1 and beta is None)):
            raise ValueError("Exactly one of alpha and beta must be None. The other one must be a float or int between 0 and 1.")
        self.n = n
        self.F = F
        self.model = Model()
        self.build(alpha, beta, outputflag)

    def build(self, alpha, beta, outputflag = 0):
        """
        Constructs the LP model by defining variables, objective, and constraints.
        
        Args:
            alpha (float or None): Consistency requirement.
            beta (float or None): Robustness requirement.
            outputflag (int): Gurobi output flag.
        
        Raises:
            ValueError: If both alpha and beta are provided or both are None.
        """
        self.model.setParam('OutputFlag', outputflag)
        self.model.setParam('MIPGap', 1e-9)

        # Add decision variables S[t, l] ∈ [0, 1]
        self.S = self.model.addVars(list(product(range(self.n+1), range(self.F.K))), lb=0, ub=1, name="S")

        # Define objective variable and set optimization direction
        if (beta is None):
            self.alpha = float(alpha)
            self.beta = self.model.addVar(name="beta", lb=0, ub=1)
            self.model.setObjective(self.beta, GRB.MAXIMIZE)
            print(f"Given alpha = {self.alpha}, maximizing beta ...")
        else:
            self.alpha = self.model.addVar(name="alpha", lb=0, ub=1)
            self.beta = float(beta)
            self.model.setObjective(self.alpha, GRB.MAXIMIZE)
            print(f"Given beta = {self.beta}, maximizing alpha ...")

        # Add consistency constraint for the prior distribution
        self.model.addConstr(self.alpha <= self.ALG(self.F.k_star, self.S), name=f"consistency_{self.F.k_star}")

        # Add robustness constraint for the prior distribution
        for k in self.F.indices:
            self.model.addConstr(self.beta <= self.ALG(k, self.S), name=f"robustness_{k}")

        # Add constraints related to survival and stopping probabilities
        for t in range(self.n+1):
            for l in range(self.F.K):
                if (t == 0): # initial condition for survival probabilities
                    self.model.addConstr(self.S[t,l] == (1 if l == self.F.K-1 else 0), name=f"Exact_{(t,l)}")
                elif (t > 1): # constraints for stopping probabilities
                    self.model.addConstr(
                        self.S[t,l] * self.F.P(t,l,"==") - self.S[t-1,l] * self.F.P(t-1,l,"==") * self.F.P(1,l+1,">=") >= 0, 
                        name=f"R_{(t,l)} <= 1"
                    )
                    self.model.addConstr(
                        self.S[t,l] * self.F.P(t,l,"==") - self.S[t-1,l] * self.F.P(t-1,l,"==") * self.F.P(1,l+1,">=") 
                        <= self.F.P(1,l,"==") * sum(self.S[t-1,m] * self.F.P(t-1,m,"==") for m in range(l, self.F.K)),
                        name=f"R_{(t,l)} >= 0"
                    )
                    
    def ALG(self, k, S):
        """
        Computes the winning probability for distribution `k`.

        The return type depends on the type of `S`:
        - If `S` is a Gurobi VarDict, returns a symbolic linear expression (`gurobipy.LinExpr`)
        - If `S` is a plain dictionary of floats, returns a numerical value (`float`)
    
        Args:
            k (int): Index of the distribution.
            S (dict or gurobipy.VarDict): Dictionary of decision variables representing survival probabilities.
    
        Returns:
            float or gurobipy.LinExpr: Winning probability or its linear expression.
        """
        c = self.F.c(k)
        return c**(self.n) * sum(
            sum(self.F.P(self.n-t,l,">=") * (
                sum(S[t-1,m] * self.F.P(t-1,m,"==") for m in range(l, self.F.K)) * self.F.P(1,l,"==")
                + S[t-1,l] * self.F.P(t-1,l,"==") * self.F.P(1,l+1,">=") - S[t,l] * self.F.P(t,l,"==")
            )
            for l in range(k, self.F.K)
                )
            for t in range(1, self.n+1)
            )
    
        
    def recover_rule(self, S):
        """
        Recovers the stopping rule from the optimized survival probabilities.
    
        Args:
            S (dict): Dictionary of optimized survival probabilities.
    
        Returns:
            dict: Mapping from (t, l) to stopping probabilities.
        """
        R = {}
        for t in range(1, self.n+1):
            for l in range(self.F.K):
                if (t == 1):
                    R[t,l] = 1 - S[t,l]
                elif (self.F.P(1,l,"==") * sum(S[t-1,m] * self.F.P(t-1,m,"==") for m in range(l, self.F.K)) > 0):
                    R[t,l] = 1 - (
                        S[t,l] * self.F.P(t,l,"==") - S[t-1,l] * self.F.P(t-1,l,"==") * self.F.P(1,l+1,">=")
                        )/(
                            self.F.P(1,l,"==") * sum(S[t-1,m] * self.F.P(t-1,m,"==") for m in range(l, self.F.K))
                        )
                else:
                    R[t,l] = None
        return R
    
    def solve(self):
        """
        Solves the LP model using Gurobi and extracts the optimized values.
    
        Returns:
            tuple:
                - S (dict): Optimized state probabilities.
                - alpha (float): Optimized or fixed consistency value.
                - beta (float): Optimized or fixed robustness value.
    
        Prints:
            Status messages and final optimized values.
    
        Returns:
            (dict, float, float): If successful.
            (None, None, None): If optimization fails or model is infeasible.
        """
        self.model.optimize()  
        status = self.model.status
        if status == GRB.OPTIMAL:
            S = {}
            for x in self.model.getVars():
                if (x.VarName.startswith('S')):
                    name = tuple(loads(x.VarName[1:]))
                    S[name] = x.X
                elif (x.VarName == 'alpha'):
                    alpha = x.X
                    beta = self.beta
                    print(f"Optimal solution found, alpha * = {alpha}")
                elif (x.VarName == 'beta'):
                    beta = x.X
                    alpha = self.alpha
                    print(f"Optimal solution found, beta * = {beta}")
            return S, alpha, beta
        elif status == GRB.INF_OR_UNBD:
            print("Model is infeasible or unbounded.")
        elif status == GRB.TIME_LIMIT:
            print("Time limit reached.")
        elif status == GRB.INTERRUPTED:
            print("Optimization was interrupted.")
        elif status == GRB.UNBOUNDED:
            print("Model is unbounded.")
        elif status == GRB.INFEASIBLE:
            print("Model is infeasible.")
        else:
            print("Optimization was not successful. Status code:", status)
        return None, None, None


def solve_single_alpha(args):
    """
    Helper function for parallel processing. Solves LP for a single alpha value.
    
    Args:
        args (tuple): Contains (alpha_id, alpha, n, p, beta)
    
    Returns:
        tuple: (alpha_id, alpha, beta_result, success)
    """
    alpha_id, alpha, n, p, beta = args
    try:
        # Suppress Gurobi output in parallel processing
        import gurobipy
        env = gurobipy.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        
        F = Distributions(p)
        model = LP_maxprob(n, F, alpha, beta, outputflag=0)
        S, a, b = model.solve()
        
        # Clean up environment
        env.dispose()
        
        if S is not None:
            return alpha_id, alpha, b, True
        else:
            return alpha_id, alpha, None, False
    except Exception:
        # Suppress error messages in parallel to avoid cluttered output
        return alpha_id, alpha, None, False


def experiment(n, p, density, simplify=False):
    """
    Runs a parametric experiment to evaluate the trade-off between consistency (alpha) 
    and robustness (beta) in a truncated distribution model.

    This function initializes a distribution and solves a linear programming model 
    for a fixed robustness level (`beta = 0`). It then iteratively varies the consistency 
    parameter `alpha` from a lower bound (0.58) to the maximum feasible value, solving 
    the model at each step and recording the resulting beta values. Only successful 
    solutions are retained and plotted.

    Args:
        n (int): Length of the arrival sequence.
        p (list of float): Probability mass function for the base distribution.
        density (float): Step size for sampling alpha values between 0.58 and the maximum feasible alpha.
        simplify (bool): If True, the program will only calculate the union of the curve and the region bounded by line segment x=0.58 and y=1/e.

    Returns:
        tuple:
            - Alpha_ (list of float): List of alpha values for which the model was successfully solved.
            - Beta_ (list of float): Corresponding beta values from the solved models.
    """
    F = Distributions(p)
    if simplify:
        model = LP_maxprob(n, F, alpha = None, beta = 1/np.e, outputflag = 0)
        _, min_alpha, _ = model.solve()
        max_alpha=0.58
    else:
        model = LP_maxprob(n, F, alpha = None, beta = 0, outputflag = 0)
        _, max_alpha, _ = model.solve()
        min_alpha=1/np.e
    if min_alpha>max_alpha:
        print("No feasible alpha-beta pair found.")
        return [], []
    print(f"alpha is experimented in [{min_alpha}, {max_alpha}]")
    Alpha = np.linspace(min_alpha, max_alpha, int((max_alpha - min_alpha)/density)+2)
    
    # Determine if we should use parallel processing
    m = len(p)
    use_parallel = n > 20 and m > 100
    
    if use_parallel:
        print(f"Using parallel processing with {mp.cpu_count()} cores for n={n}, m={m}")
        print(f"Processing {len(Alpha)} alpha values...")
        
        # Prepare arguments for parallel processing
        args_list = [(alpha_id, alpha, n, p, None) for alpha_id, alpha in enumerate(Alpha)]
        
        # Use multiprocessing Pool with progress tracking
        Alpha_ = []
        Beta_ = []
        
        # Process in chunks to provide progress updates
        chunk_size = max(1, len(args_list) // (mp.cpu_count() * 4))  # 4 chunks per core
        
        with mp.Pool(processes=mp.cpu_count()) as pool:
            result=pool.map(solve_single_alpha, args_list, chunksize=chunk_size)
        
        print(result)
        Alpha_=[0]*len(result)
        Beta_=[0]*len(result)
        for alpha_id, alpha, beta_result, success in result:
            if success:
                Alpha_[alpha_id] = alpha
                Beta_[alpha_id] = beta_result
        
        print(f"Parallel processing completed. {len(Alpha_)} successful solutions out of {len(Alpha)} total.")
    else:
        print(f"Using sequential processing for n={n}, m={m}")
        Beta = []
        solved_set = []
        beta = None
        for alpha_id, alpha in enumerate(Alpha):
            model = LP_maxprob(n, F, alpha, beta, outputflag=0)
            S, a, b = model.solve()
            Beta.append(b)
            if S is not None:
                solved_set.append(alpha_id)
        Alpha_ = [alpha for i, alpha in enumerate(Alpha) if i in solved_set]
        Beta_ = [beta for i, beta in enumerate(Beta) if i in solved_set]
    
    return Alpha_, Beta_


class alpha_beta_curve:
    """
    Two ways to initialize:
    - Run experiment: provide n and p (and optional simplify/density).
    - Load from file: provide datafile path (tab-separated with optional comment lines starting with '#').

    After initialization the instance has attributes:
        - Alpha: list of alpha values
        - Beta: list of beta values
        - n, p: set when created by experiment, otherwise None
    """
    def __init__(self, n=None, m=None, simplify=False, density=0.001, load_from_file=True, suppfunc=lambda x: 1/x, strp=f"1%x"):
        if n is None or m is None:
            raise ValueError("When datafile is not provided, both n and m must be given.")
        self.n = n
        self.m = m
        self.p = [suppfunc(i) for i in range(1, m+1)]
        self.strp=strp
        self.simplify = simplify
        # construct alpha-beta datalist
        if load_from_file is True:
            # try to load from file
            try:
                if self.simplify:
                    strsimpl="simplified"
                else:
                    strsimpl="full"
                datafile=os.path.join(outdir,strsimpl, f"n={n},m={m}, p={strp}.txt")
                self.Alpha, self.Beta= self._load_from_file(datafile)
            except Exception as e:
                raise ValueError(f"Failed to load data from file {datafile}: {e}")
        else:
            print(f"An experiment with n={n}, m={m}, p={strp}, density={density} is running ...")
            self.Alpha, self.Beta = experiment(self.n, self.p, density, simplify=simplify)
            print("Experiment finished.")
            self.save()

    def _load_from_file(self, path):
        Alpha = []
        Beta = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    a = float(parts[0])
                    b = float(parts[1])
                except ValueError:
                    continue
                Alpha.append(a)
                Beta.append(b)
        return Alpha, Beta

    def save(self, filename=None):
        """
        Save this instance's Alpha/Beta to a file in the same format used elsewhere.
        If filename is None a default name is constructed from n,m,safe_strp.
        """
        os.makedirs(outdir, exist_ok=True)
        if filename is None:
            filename = f"n={self.n},m={self.m}, p={self.strp}.txt"
        if self.simplify:
            strsimpl="simplified"
        else:
            strsimpl="full"
        outpath = os.path.join(outdir,strsimpl, filename)
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        with open(outpath, "w") as f:
            f.write("# alpha\tbeta\n")
            for a, b in zip(self.Alpha, self.Beta):
                f.write(f"{a:.12g}\t{b:.12g}\n")

    def plot(self,savefig=True,xaxisrange=(0,1),yaxisrange=(0,1),add_legend=False):
        # plot
        # plot main curve with label if provided (otherwise use a default)
        if not add_legend:
            plt.plot(self.Alpha, self.Beta)
        else:
            plt.plot(self.Alpha, self.Beta, label="our hardness curve")
        # plot trivial hardness curve with its own label
        if not add_legend:
            plt.plot([0.58, 0.58, np.exp(-1)], [0, np.exp(-1), np.exp(-1)], '-')
        else:
            plt.plot([0.58, 0.58, np.exp(-1)], [0, np.exp(-1), np.exp(-1)], '-', label="trivial hardness")
        if add_legend:
            plt.legend()

        # lock axes to [0,1] x [0,1]
        plt.xlim(xaxisrange)
        plt.ylim(yaxisrange)

        # prepare filename and directory, sanitize p -> strp
        os.makedirs(outdir, exist_ok=True)
        # sanitize strp to avoid filesystem separators in filename (e.g. "1/x^3" -> "1%x^3")
        filename = f"n={self.n},m={self.m}, p={self.strp}.png"
        if self.simplify:
            strsimpl="simplified"
        else:
            strsimpl="full"
        outpath = os.path.join(outdir, strsimpl, filename)

        # save figure
        if savefig:
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            plt.savefig(outpath, bbox_inches="tight")

if __name__ == '__main__':
    # Example usage
    n = 24  # number of arrivals
    density = 0.001  # density: step size for alpha sampling
    strp = "1%x" # strp: the string representation of the probability mass function for file naming
    for m in [160]: # a list of m (support size) to try
        # simplify = False: full curve over [0,1]x[0,1]
        # simplify = True: only the part bounded by x=0.58 and y=1/e
        # load_from_file=False: run experiment
        # True: load from file if exists
        # The data will automatically be saved to a folder named output in the same directory after experiment.
        # suppfunc: the function to generate the probability mass function
        data = alpha_beta_curve(
            n=n, m=m, simplify=False, density=density,
            load_from_file=True, suppfunc=lambda x: 1/x, strp=strp
        )
        xaxisrange = (0, 0.8)
        yaxisrange = (0, 0.6)
        # Plot the constructed hardness curve against the trivial hardness curve
        data.plot(xaxisrange=xaxisrange,yaxisrange=yaxisrange,add_legend=True)

    plt.show()

# Given beta = 0.36787944117144233, maximizing alpha ...
# Optimal solution found, alpha * = 0.5602568817156317
# alpha is experimented in [0.5602568817156317, 0.58]
# Given alpha = 0.5602568817156317, maximizing beta ...
# Optimal solution found, beta * = 0.3678794464707393