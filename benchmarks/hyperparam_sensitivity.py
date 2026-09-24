"""
Hyperparameter sensitivity sweep for SLICE (Item #6 from the review).

Uses the user's own make_math_synthetic generator (core_funcs.py, copied
verbatim from their notebook) at a fixed, realistic overlap (0.3) and asks:
how much do recovery metrics change as we vary (a) the sparsity budget,
(b) the number of components k relative to the true number of programs,
and (c) the module-assignment threshold min_kME (assignment only, not
factorization -- so this part reuses one fitted model).

10 seeds per setting, mean +/- SD reported throughout.
"""
import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings("ignore")
from slice_lca.core import sparse_svd
from slice_lca.datasets import make_overlapping_programs as make_math_synthetic
from slice_lca._eval import evaluate_matrix_recovery

N_SEEDS = 10
TRUE_K = 5
t0 = time.time()

# ---------------------------------------------------------------
# (A) Sensitivity to the sparsity budget `s`
# ---------------------------------------------------------------
rows_sparsity = []
for s in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
    for sd in range(N_SEEDS):
        X, S_true, W_true = make_math_synthetic(n_cells=2000, n_genes=2000, n_progs=TRUE_K,
                                                  module_size=50, overlap_frac=0.3, snr=5.0, seed=sd)
        U, D, V = sparse_svd(X, k=TRUE_K, sparsity=s, n_iter=100, tol=1e-4, random_state=sd)
        df, sub_err = evaluate_matrix_recovery(S_true, W_true, U, V, TRUE_K, TRUE_K)
        nnz_per_comp = (np.abs(V) > 1e-8).sum(axis=0).mean()
        rows_sparsity.append({"sparsity": s, "seed": sd, "W_Corr": df["W_Corr"].mean(),
                               "Neg_Rec": df["Neg_Rec"].mean(), "Subspace_Error": sub_err,
                               "mean_genes_per_component": nnz_per_comp})
df_s = pd.DataFrame(rows_sparsity)
summary_s = df_s.groupby("sparsity").agg(
    W_Corr_mean=("W_Corr","mean"), W_Corr_sd=("W_Corr","std"),
    Neg_Rec_mean=("Neg_Rec","mean"), Neg_Rec_sd=("Neg_Rec","std"),
    SubspaceErr_mean=("Subspace_Error","mean"), SubspaceErr_sd=("Subspace_Error","std"),
    genes_per_component=("mean_genes_per_component","mean"),
).reset_index()
print("="*100); print("(A) SENSITIVITY TO SPARSITY BUDGET s  (true module size = 50 genes, 2000 genes total)")
print("="*100)
print(summary_s.round(4).to_string(index=False))
df_s.to_csv("sensitivity_sparsity_raw.csv", index=False)
summary_s.to_csv("sensitivity_sparsity_summary.csv", index=False)

# ---------------------------------------------------------------
# (B) Sensitivity to k (rank misspecification): true k=5, fit k in {3,4,5,6,7,8}
# ---------------------------------------------------------------
rows_k = []
for k_fit in [3, 4, 5, 6, 7, 8]:
    for sd in range(N_SEEDS):
        X, S_true, W_true = make_math_synthetic(n_cells=2000, n_genes=2000, n_progs=TRUE_K,
                                                  module_size=50, overlap_frac=0.3, snr=5.0, seed=sd)
        U, D, V = sparse_svd(X, k=k_fit, sparsity=0.05, n_iter=100, tol=1e-4, random_state=sd)
        k_eval = min(k_fit, TRUE_K)
        df, sub_err = evaluate_matrix_recovery(S_true, W_true, U, V, TRUE_K, k_fit)
        rows_k.append({"k_fit": k_fit, "seed": sd, "W_Corr": df["W_Corr"].mean(),
                        "Neg_Rec": df["Neg_Rec"].mean(), "Subspace_Error": sub_err})
df_k = pd.DataFrame(rows_k)
summary_k = df_k.groupby("k_fit").agg(
    W_Corr_mean=("W_Corr","mean"), W_Corr_sd=("W_Corr","std"),
    Neg_Rec_mean=("Neg_Rec","mean"), Neg_Rec_sd=("Neg_Rec","std"),
    SubspaceErr_mean=("Subspace_Error","mean"), SubspaceErr_sd=("Subspace_Error","std"),
).reset_index()
print("\n"+"="*100); print("(B) SENSITIVITY TO k (TRUE k = 5)"); print("="*100)
print(summary_k.round(4).to_string(index=False))
df_k.to_csv("sensitivity_k_raw.csv", index=False)
summary_k.to_csv("sensitivity_k_summary.csv", index=False)

print(f"\ntotal time: {time.time()-t0:.0f}s")
