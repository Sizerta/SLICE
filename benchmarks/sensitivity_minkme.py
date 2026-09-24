"""
(C) Sensitivity to the module-assignment threshold min_kME (theta).
Fits SLICE ONCE per seed (sparsity=0.02, k=5, true k=5), then sweeps
the assignment threshold on the SAME fitted U to see how the fraction
of assigned ("non-grey") genes changes, and whether genes belonging to
a true program keep getting assigned to *some* module as theta varies
(assignment stability), separate from factorization quality itself.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from slice_lca.core import sparse_svd
from slice_lca.datasets import make_overlapping_programs as make_math_synthetic

N_SEEDS = 10
TRUE_K = 5
thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
rows = []

for sd in range(N_SEEDS):
    X, S_true, W_true = make_math_synthetic(n_cells=2000, n_genes=2000, n_progs=TRUE_K,
                                              module_size=50, overlap_frac=0.3, snr=5.0, seed=sd)
    U, D, V = sparse_svd(X, k=TRUE_K, sparsity=0.02, n_iter=100, tol=1e-4, random_state=sd)
    # kME = correlation of each gene's expression with each module eigengene (U column)
    Xd = X.toarray()
    kME = np.zeros((Xd.shape[1], TRUE_K))
    for m in range(TRUE_K):
        for j in range(Xd.shape[1]):
            kME[j, m] = np.corrcoef(Xd[:, j], U[:, m])[0, 1]
    true_module = (np.abs(W_true) > 1e-8).any(axis=1)  # genes that truly belong to some program
    for th in thresholds:
        best_mod = np.argmax(np.abs(kME), axis=1)
        best_val = kME[np.arange(kME.shape[0]), best_mod]
        assigned = np.abs(best_val) >= th
        frac_assigned = assigned.mean()
        # of the genes that are TRULY in a program, what fraction get assigned (recall)?
        recall_true_genes = assigned[true_module].mean()
        # of assigned genes, what fraction are truly in a program (precision)?
        precision = true_module[assigned].mean() if assigned.sum() > 0 else np.nan
        rows.append({"seed": sd, "min_kME": th, "frac_genes_assigned": frac_assigned,
                     "recall_on_true_module_genes": recall_true_genes, "precision": precision})

df = pd.DataFrame(rows)
summary = df.groupby("min_kME").agg(
    frac_assigned_mean=("frac_genes_assigned","mean"), frac_assigned_sd=("frac_genes_assigned","std"),
    recall_mean=("recall_on_true_module_genes","mean"), recall_sd=("recall_on_true_module_genes","std"),
    precision_mean=("precision","mean"), precision_sd=("precision","std"),
).reset_index()
print("="*100); print("(C) SENSITIVITY TO min_kME ASSIGNMENT THRESHOLD (theta)"); print("="*100)
print(summary.round(4).to_string(index=False))
df.to_csv("sensitivity_minkme_raw.csv", index=False)
summary.to_csv("sensitivity_minkme_summary.csv", index=False)
