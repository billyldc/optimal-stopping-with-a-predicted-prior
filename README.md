# Optimal Stopping with a Predicted Prior - Code Repository

This repository contains code to reproduce numerical experiments and plots from the paper "Optimal Stopping with a Predicted Prior" and focuses on two main objectives: (1) maximizing expected value (MaxExp) and (2) maximizing the probability of selecting the maximum (MaxProb).

## Contents
### MaxExp setting (Theorem 1)
The following files are included:
- `MaxExp.py` — plotting driver for Figure 1(a), which illustrates the results in Theorem 1 (the MaxExp trade-off). Loads `Algorithm_MaxExp.txt` and draws:
	- the baseline algorithmic curve as the linear interpolation between Dynkin's algorithm (1/e, 1/e) and the optimal i.i.d. prophet inequality algorithm under the MaxExp setting proposed by Hill and Kertz (1982), $(\alpha^*_{\text{MaxExp}} = 0.745, 0)$
	- the algorithmic curve verified by `Algorithm_MaxExp.py` and the tangent interpolation with $(\alpha^*_{\text{MaxExp}}, 0)$
	- the baseline hardness lines implied by hardness results from the literature (1/e hardness from Dynkin (1963) and $\alpha^*_{\text{MaxExp}}$ hardness from Hill and Kertz (1982))

- `Algorithm_MaxExp.py` — Related to Appendix B. Verifies the (α, β) values stored in `Algorithm_MaxExp.txt` are feasible by recursively constructing a stepwise threshold function using the procedure described in Appendix B. It also draws Fig. 3 with α = 0.6908 and β = 0.01. The verification results (the stepwise threshold function) are stored in `valid_threshold_functions.txt`.

### MaxProb setting (Theorems 2 and 3)

- `MaxProb.py` — plotting driver for Figure 1(b), which illustrates the results in Theorems 2 and 3 (the MaxProb trade-off). Loads `Algorithm_MaxProb.txt` and draws:
	- the baseline algorithmic curve as the linear interpolation between Dynkin's algorithm (1/e, 1/e) and the optimal i.i.d. prophet inequality algorithm under the MaxProb setting, $(\alpha^*_{\text{MaxProb}} = 0.580, 0)$
	- the algorithmic curve computed by `Algorithm_MaxProb.py`
	- the baseline hardness lines implied by hardness results from the literature (1/e hardness from Dynkin (1963) and $\alpha^*_{\text{MaxProb}}$ hardness from Gilbert and Mosteller (1966))
	- the hardness curve computed by `Hardness_MaxProb.py` by solving a linear program, as discussed in Section 8

- `Algorithm_MaxProb.py` — Related to Theorem 2. Numerically computes the algorithmic (α, β) curve for MaxProb by evaluating the double integral proposed in Theorem 2; saves to and reads from `Algorithm_MaxProb.txt`.

- `Hardness_MaxProb.py` — Related to Theorem 3. Builds and solves parameterized linear programs to obtain hardness (lower-bound) trade-off points (α, β) for MaxProb. Uses Gurobi to solve LPs over a discretized family of distributions. The results for n=30, K=1024 correspond to `Hardness_MaxProb_n=30_K=1024.txt`.

### Utilities and data
- `helper.py` — shared utilities: root/quad wrappers, I/O helpers (`save_data`, `read_data`), plotting helpers (curves, baselines, formatting), and analytic root-finding helpers (`compute_λ`, `compute_α_star_MaxExp`).
- `Hardness_MaxProb_n=30_K=1024.txt` — precomputed hardness curve for MaxProb with n=30, K=1024.
- `Algorithm_MaxExp.txt`, `Algorithm_MaxProb.txt` — saved numeric (α, β) pairs used by the plotting scripts.
- `valid_threshold_functions.txt` — output produced by `Algorithm_MaxExp.py` containing the computed step-function thresholds for feasible (α, β) pairs.
- `LICENSE`, `.gitignore`, auxiliary `.txt`/`.py` files.

## Dependencies
- Python 3.8+ (uses f-strings and modern SciPy/NumPy APIs)
- numpy, scipy, matplotlib
- gurobipy (Gurobi Python API) for `Hardness_MaxProb.py` (optional if you only plot precomputed hardness files)
