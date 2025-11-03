import numpy as np
from itertools import product
from gurobipy import *
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
from helper import save_data, read_data, plot_tradeoff_curve
from Algorithm_MaxProb import solve_γ, compute_α_for_MaxProb


class LPsolver_MaxProb:

    def __init__(self, n, masses, λ, env=None):

        if not all(m >= 0 for m in masses):
            raise ValueError("All relative masses must be nonnegative.")
        masses = [m / sum(masses) for m in masses]
        self.f = {l: masses[l - 1] for l in range(1, len(masses) + 1)}
        self.K = len(self.f)
        self.n = n
        if env is not None:
            self.model = Model(env=env)
        else:
            self.model = Model()
        self.model.Params.OutputFlag = 0
        self.build(λ)

    def F(self, l):
        return np.sum([self.f[i] for i in range(1, l + 1)])

    def Δ(self, t, l):
        if t == 0:
            return 1 if l == 1 else 0
        return self.F(l) ** t - self.F(l - 1) ** t

    def build(self, λ):

        self.model.setParam("MIPGap", 1e-9)

        self.y = self.model.addVars(
            list(product(range(self.n + 1), range(1, self.K + 1))), lb=0, ub=1, name="y"
        )
        self.α = self.model.addVar(name="α", lb=0, ub=1)
        self.β = self.model.addVar(name="β", lb=0, ub=1)
        self.model.setObjective(λ*self.α + (1-λ)*self.β, GRB.MAXIMIZE)

        for l in range(1, self.K + 1):
            self.model.addConstr(self.y[0, l] == 1, name=f"REJ_{(0,l)} == 1")

        S = 0
        for l in range(1, self.K + 1):
            for t in range(1, self.n + 1):
                A = self.Δ(t, l) * self.y[t, l]
                B = self.Δ(t - 1, l) * self.F(l - 1) * self.y[t - 1, l]
                C = sum(
                    self.Δ(t - 1, m) * self.f[l] * self.y[t - 1, m]
                    for m in range(1, l + 1)
                )
                self.model.addConstr(A - B >= 0, name=f"ACC_{(t,l)} <= 1")
                self.model.addConstr(A - B <= C, name=f"ACC_{(t,l)} >= 0")
                S = S + self.F(l) ** (self.n - t) * (C + B - A)
            self.model.addConstr(
                S / self.F(l) ** self.n >= self.β, name=(f"robustness_{l}")
            )
            if l == self.K:
                self.model.addConstr(
                    S / self.F(l) ** self.n >= self.α, name=(f"consistency_{l}")
                )

    def solve(self):
        self.model.optimize()
        y = {}
        for x in self.model.getVars():
            if x.VarName == "α":
                α = x.X
            elif x.VarName == "β":
                β = x.X
        return α, β


_worker_env = None


def init_worker():
    global _worker_env
    _worker_env = Env(empty=True)
    _worker_env.setParam("OutputFlag", 0)
    _worker_env.start()


def evaluate_λ_point(args):
    global _worker_env
    n, masses, λ = args
    model = LPsolver_MaxProb(n, masses, λ, env=_worker_env)
    α, β = model.solve()
    return λ, α, β


def compute_tradeoff_curve(n, masses, num_points, filename=None):

    args_list = [(n, masses, i/num_points) for i in range(num_points+1) if i%5!=0]

    n_tasks = len(args_list)
    n_workers = 3
    split = [i for i in range(28, n_tasks, 3)]
    result = []
    for i in range(len(split)):
        with mp.Pool(processes=n_workers, initializer=init_worker) as pool:
            partial = pool.map(evaluate_λ_point, args_list[split[i]:(split[i+1] if i+1 <= len(split)-1 else n_tasks)])
        print(partial)
        result.extend(partial)
        save_data(np.array(partial), f"output_continue/lambda_Hardness_MaxProb_n={n}_K={K}_i={split[i]}_to_{(split[i+1] if i+1 <= len(split)-1 else n_tasks)}.txt")

    if filename is not None:
        arr = np.array(result)
        save_data(arr, filename, header = "λ α β")

    _, α_values, β_values = zip(*[(λ, α, β) for λ, α, β in result if α is not None])

    return α_values, β_values


def plot_hardness_MaxProb(ax, n, K, num_points = 100, color="tab:green", filename=None):
    if not os.path.exists(filename):
        masses = [1 / k for k in range(1, K + 1)]
        α_values, β_values = compute_tradeoff_curve(
            n, masses, num_points, filename=filename
        )
    else:
        α_values, β_values = read_data(filename)
    plot_tradeoff_curve(
        ax, α_values, β_values, mode="hard", color=color, label="Our hardness"
    )


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=500)
    n, K = 3,3#30, 1024
    plot_hardness_MaxProb(
        ax, n, K, num_points = 100,
        filename=f"lambda_Hardness_MaxProb_n={n}_K={K}.txt"
    )
    plt.tight_layout()
    plt.legend()
    plt.show()
