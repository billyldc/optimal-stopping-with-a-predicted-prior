import numpy as np
from itertools import product
from gurobipy import *
from json import loads
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import sys
from helper import save_data, read_data, plot_tradeoff_curve
from Algorithm_MaxProb import solve_γ, compute_α_for_MaxProb


class LPsolver_MaxProb:

    def __init__(self, n, masses, α=None, β=None, env=None):

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
        self.build(α, β)

    def F(self, l):
        return np.sum([self.f[i] for i in range(1, l + 1)])

    def Δ(self, t, l):
        if t == 0:
            return 1 if l == 1 else 0
        return self.F(l) ** t - self.F(l - 1) ** t

    def build(self, α, β):

        self.model.setParam("MIPGap", 1e-9)

        self.y = self.model.addVars(
            list(product(range(self.n + 1), range(1, self.K + 1))), lb=0, ub=1, name="y"
        )
        self.α = self.model.addVar(name="α", lb=0, ub=1)
        self.β = self.model.addVar(name="β", lb=0, ub=1)
        if α is not None:
            self.model.setObjective(self.β, GRB.MAXIMIZE)
            self.model.addConstr(self.α >= α, name="α_lower_bound")
        elif β is not None:
            self.model.setObjective(self.α, GRB.MAXIMIZE)
            self.model.addConstr(self.β >= β, name="β_lower_bound")

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


def evaluate_alpha_point(args):
    global _worker_env
    n, masses, α = args
    model = LPsolver_MaxProb(n, masses, α=α, β=None, env=_worker_env)
    α, β = model.solve()
    return α, β


def compute_tradeoff_curve(n, masses, density, filename=None):
    min_α, _ = LPsolver_MaxProb(n, masses, α=None, β=1 / np.e).solve()
    γ = solve_γ()
    max_α = compute_α_for_MaxProb(0, γ)

    if min_α > max_α:
        return [], []

    α_values = np.linspace(min_α, max_α, int((max_α - min_α) / density) + 2)

    args_list = [(n, masses, α) for α in α_values]

    n_tasks = len(args_list)
    n_workers = min(mp.cpu_count(), n_tasks)
    with mp.Pool(processes=n_workers, initializer=init_worker) as pool:
        result = pool.map(evaluate_alpha_point, args_list)

    if filename is not None:
        save_data(result, filename)

    α_values, β_values = zip(*[(α, β) for α, β in result if α is not None])

    return α_values, β_values


def plot_hardness_MaxProb(ax, n, K, density=0.001, color="tab:green", filename=None):
    if not os.path.exists(filename):
        n = 20
        K = 256
        masses = [1 / k for k in range(1, K + 1)]
        α_values, β_values = compute_tradeoff_curve(
            n, masses, density, filename=filename
        )
    else:
        α_values, β_values = read_data(filename)
    plot_tradeoff_curve(
        ax, α_values, β_values, mode="hard", color=color, label="Our hardness"
    )


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=500)
    n, K = 20, 256
    plot_hardness_MaxProb(
        ax, n, K, filename="../max_prob/output/simplified/n=30,m=1024, p=1%x.txt"
    )
    plt.tight_layout()
    plt.legend()
    plt.show()
