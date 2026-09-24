"""
Validation for sparse_svd_multistart: does best-of-N-restarts actually
help, and does the effect replicate on seeds that weren't used to find
it?

Two runs on a deliberately hard synthetic regime (more programs than
the paper's default examples, high overlap, low SNR -- chosen to make
bad-local-optimum outcomes more likely so any real effect is visible):
  1. Seeds 0-24 (where the effect was first observed)
  2. Seeds 1000-1014 (completely disjoint range, run afterward, to
     rule out the first result being an artifact of those specific
     seeds)

Both should show the same direction and rough magnitude of effect.
"""
import time
import warnings

import numpy as np
import pandas as pd

from slice_lca.core import sparse_svd, sparse_svd_multistart
from slice_lca._eval import evaluate_matrix_recovery
from slice_lca.datasets import make_overlapping_programs

warnings.filterwarnings("ignore")

N_PROGS = 8
X, S_true, W_true = make_overlapping_programs(
    n_cells=2000, n_genes=2000, n_progs=N_PROGS, module_size=50,
    overlap_frac=0.6, snr=1.5, seed=0,
)


def run(seed_range, label):
    t0 = time.time()
    rows = []
    for seed in seed_range:
        U0, D0, V0 = sparse_svd(X, k=N_PROGS, sparsity=0.05, n_iter=60, tol=1e-4, random_state=seed)
        df0, sub0 = evaluate_matrix_recovery(S_true, W_true, U0, V0, N_PROGS, N_PROGS)
        rows.append({"mode": "single_start", "seed": seed, "W_Corr": df0["W_Corr"].mean(), "Subspace_Error": sub0})

        U1, D1, V1, best_seed, scores = sparse_svd_multistart(
            X, k=N_PROGS, n_restarts=4, sparsity=0.05, n_iter=60, tol=1e-4, random_state=seed)
        df1, sub1 = evaluate_matrix_recovery(S_true, W_true, U1, V1, N_PROGS, N_PROGS)
        rows.append({"mode": "multistart_4", "seed": seed, "W_Corr": df1["W_Corr"].mean(), "Subspace_Error": sub1})

    df = pd.DataFrame(rows)
    print(f"=== {label} ({time.time()-t0:.0f}s) ===")
    print(df.groupby("mode")[["W_Corr", "Subspace_Error"]].agg(["mean", "min", "std"]).round(4))
    print()
    return df


df_discovery = run(range(25), "discovery run, seeds 0-24")
df_holdout = run(range(1000, 1015), "held-out confirmation, seeds 1000-1014")

pd.concat([df_discovery.assign(split="discovery"), df_holdout.assign(split="holdout")]).to_csv(
    "mathcore_multistart_validation_raw.csv", index=False
)
