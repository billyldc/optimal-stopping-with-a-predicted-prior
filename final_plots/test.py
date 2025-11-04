import os
import re
import numpy as np
from helper import read_data,save_data
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

save_data(np.column_stack((alphas,betas)),os.path.join(os.path.dirname(__file__), "alpha_betas_np.txt"))

