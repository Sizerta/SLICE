"""
Evaluation metrics used by the paper's benchmark scripts (`benchmarks/`).
Kept separate from `slice_lca.core` because these are benchmarking
utilities, not part of the library's public API.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA, SparsePCA, NMF
from sklearn.metrics import mean_squared_error

from .core import sparse_svd
from .datasets import make_overlapping_programs, make_antagonistic_axis


def calc_subspace_error(W_true, W_pred):
    U_t, _, _ = np.linalg.svd(W_true, full_matrices=False)
    U_p, _, _ = np.linalg.svd(W_pred, full_matrices=False)
    P_true = U_t @ U_t.T
    P_pred = U_p @ U_p.T
    return np.linalg.norm(P_true - P_pred, "fro")


def evaluate_matrix_recovery(S_true, W_true, U_pred, V_pred, k_true, k_pred):
    corr_matrix = np.zeros((k_true, k_pred))
    for i in range(k_true):
        for j in range(k_pred):
            corr_matrix[i, j] = np.abs(np.corrcoef(W_true[:, i], V_pred[:, j])[0, 1])
    row_ind, col_ind = linear_sum_assignment(-corr_matrix)
    metrics = []
    for i, j in zip(row_ind, col_ind):
        w_true, w_pred = W_true[:, i], V_pred[:, j]
        s_true, s_pred = S_true[:, i], U_pred[:, j]
        if np.corrcoef(w_true, w_pred)[0, 1] < 0:
            w_pred, s_pred = -w_pred, -s_pred
        w_corr = np.corrcoef(w_true, w_pred)[0, 1]
        s_corr = np.corrcoef(s_true, s_pred)[0, 1]
        true_pos, true_neg = w_true > 0.1, w_true < -0.1
        pred_pos, pred_neg = w_pred > 0.1, w_pred < -0.1
        pos_prec = np.sum(true_pos & pred_pos) / (np.sum(pred_pos) + 1e-9)
        pos_rec = np.sum(true_pos & pred_pos) / (np.sum(true_pos) + 1e-9)
        neg_prec = np.sum(true_neg & pred_neg) / (np.sum(pred_neg) + 1e-9)
        neg_rec = np.sum(true_neg & pred_neg) / (np.sum(true_neg) + 1e-9)
        metrics.append({
            "True_Prog": i, "Pred_Prog": j, "W_Corr": w_corr, "S_Corr": s_corr,
            "Pos_Prec": pos_prec, "Pos_Rec": pos_rec, "Neg_Prec": neg_prec, "Neg_Rec": neg_rec,
        })
    df_metrics = pd.DataFrame(metrics)
    subspace_err = calc_subspace_error(W_true, V_pred[:, [m["Pred_Prog"] for m in metrics]])
    return df_metrics, subspace_err


def benchmark_methods_comparison(overlap_frac=0.2, snr=5.0, seed=42):
    n_progs = 5
    module_size = 50
    n_genes = 2000
    X, S_true, W_true = make_overlapping_programs(
        n_cells=2000, n_genes=n_genes, n_progs=n_progs,
        module_size=module_size, overlap_frac=overlap_frac, snr=snr, seed=seed,
    )
    results = []
    X_dense = X.toarray() if sp.issparse(X) else X

    U_slice, D_slice, V_slice = sparse_svd(X, k=n_progs, sparsity=0.05, n_iter=100, tol=1e-4, random_state=seed)
    df_slice, sub_err_slice = evaluate_matrix_recovery(S_true, W_true, U_slice, V_slice, n_progs, n_progs)
    results.append({"Method": "SLICE", "Avg_W_Corr": df_slice["W_Corr"].mean(),
                     "Avg_Neg_Recall": df_slice["Neg_Rec"].mean(), "Subspace_Error": sub_err_slice})

    spca = SparsePCA(n_components=n_progs, alpha=1.0, random_state=seed, max_iter=100)
    U_spca = spca.fit_transform(X_dense)
    V_spca = spca.components_.T
    df_spca, sub_err_spca = evaluate_matrix_recovery(S_true, W_true, U_spca, V_spca, n_progs, n_progs)
    results.append({"Method": "Sparse PCA", "Avg_W_Corr": df_spca["W_Corr"].mean(),
                     "Avg_Neg_Recall": df_spca["Neg_Rec"].mean(), "Subspace_Error": sub_err_spca})

    pca = PCA(n_components=n_progs, random_state=seed)
    U_pca = pca.fit_transform(X_dense)
    V_pca = pca.components_.T
    df_pca, sub_err_pca = evaluate_matrix_recovery(S_true, W_true, U_pca, V_pca, n_progs, n_progs)
    results.append({"Method": "PCA", "Avg_W_Corr": df_pca["W_Corr"].mean(),
                     "Avg_Neg_Recall": df_pca["Neg_Rec"].mean(), "Subspace_Error": sub_err_pca})

    X_nmf = np.maximum(X_dense, 0)
    nmf = NMF(n_components=n_progs, init="nndsvda", random_state=seed, max_iter=200)
    U_nmf = nmf.fit_transform(X_nmf)
    V_nmf = nmf.components_.T
    df_nmf, sub_err_nmf = evaluate_matrix_recovery(S_true, W_true, U_nmf, V_nmf, n_progs, n_progs)
    results.append({"Method": "NMF", "Avg_W_Corr": df_nmf["W_Corr"].mean(),
                     "Avg_Neg_Recall": df_nmf["Neg_Rec"].mean(), "Subspace_Error": sub_err_nmf})

    return pd.DataFrame(results)


def evaluate_antagonistic_efficiency(X, S_true, W_true, nmf_seed=42):
    X_dense = X.toarray() if sp.issparse(X) else X
    results = []
    U_slice, D_slice, V_slice = sparse_svd(X, k=1, sparsity=0.15, n_iter=100, tol=1e-4, random_state=0)
    if np.corrcoef(W_true[:, 0], V_slice[:, 0])[0, 1] < 0:
        V_slice[:, 0], U_slice[:, 0] = -V_slice[:, 0], -U_slice[:, 0]
    slice_rec = mean_squared_error(X_dense, (U_slice * D_slice[0]) @ V_slice.T)
    slice_s_corr = np.abs(np.corrcoef(S_true[:, 0], U_slice[:, 0])[0, 1])
    slice_w_corr = np.abs(np.corrcoef(W_true[:, 0], V_slice[:, 0])[0, 1])
    results.append({"Method": "SLICE", "k": 1, "MSE": slice_rec, "State_Corr": slice_s_corr, "Gene_Corr": slice_w_corr})
    for k_nmf in [1, 2, 3]:
        X_nmf = np.maximum(X_dense, 0)
        nmf = NMF(n_components=k_nmf, init="nndsvda", random_state=nmf_seed, max_iter=500)
        U_nmf = nmf.fit_transform(X_nmf)
        V_nmf = nmf.components_.T
        nmf_rec = mean_squared_error(X_dense, U_nmf @ V_nmf.T)
        best_s_corr = max(np.abs(np.corrcoef(S_true[:, 0], U_nmf[:, i])[0, 1]) for i in range(k_nmf))
        results.append({"Method": "NMF", "k": k_nmf, "MSE": nmf_rec, "State_Corr": best_s_corr, "Gene_Corr": np.nan})
    return pd.DataFrame(results)
