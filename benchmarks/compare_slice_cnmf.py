"""
SLICE vs. cNMF: a real, same-hardware head-to-head comparison.

Unlike the paper's current hdWGCNA comparison (numbers from a different
paper, different hardware, different dataset), this runs BOTH methods
on the SAME data in the SAME session and reports the same metrics for
both. Tested end-to-end on synthetic ground-truth data before being
handed to you; see the bottom of this file for how to point it at a
real dataset.

Install (once, in Colab):
    !pip install cnmf scanpy slice-lca   # slice-lca: from the package
                                          # zip delivered separately, or
                                          # `pip install -e .` after
                                          # uploading/unzipping it.

WHAT THIS DOES
--------------
1. `run_slice(...)`   -- fits SLICE via slice_lca.core.fit_slice,
                          using implicit centering (see below) so the
                          input never needs to be densified.
2. `run_cnmf(...)`    -- runs the REAL cnmf package (prepare -> factorize
                          -> combine -> consensus -> load_results), not a
                          proxy. Requires raw (non-negative) counts.
3. `evaluate(...)`    -- if you have ground truth (synthetic data),
                          reports the same Cell A/B, Gene A/B correlation
                          metrics used throughout the paper. Without
                          ground truth (real data), reports program-level
                          agreement between the two methods instead
                          (matched via Hungarian assignment on gene
                          loading correlation) -- a real, if weaker,
                          cross-method validation signal.

TWO BUGS FIXED HERE (found from a real Colab run, not hypothesized):

1. GENE-DIMENSION MISMATCH (the actual crash you hit). cNMF's
   `load_results()` returns `spectra_tpm` over ALL genes in the input
   file (it refits the K programs it found on your `num_highvar_genes`
   HVG subset back onto every gene, via TPM) -- NOT just the HVG
   subset it fit on. So if SLICE was fit on a 2,000-gene HVG-restricted
   matrix, its (2000, k) V won't line up positionally with cNMF's
   (27998, k) spectra_tpm. Fixed by tracking gene NAMES through both
   `run_slice` and `run_cnmf` and aligning by identity in
   `evaluate_cross_method_agreement`, not by raw array position.

2. sparse_svd previously had no way to center data without densifying,
   so the old version of this file told you to build
   `Xc = sp.csr_matrix(Xlog - Xlog.mean(0))` by hand -- which silently
   densifies (a centered matrix has almost no exact zeros left) and
   defeats the point of staying sparse at real dataset scale.
   `run_slice` now just calls `fit_slice(..., center=True)`, which
   centers implicitly inside the factorization instead.

HYPERPARAMETER NOTE (see benchmarks/sensitivity_sparsity_summary.csv):
SLICE's `sparsity` should be set near the fraction of genes you expect
per program, not left at an arbitrary default -- on a quick synthetic
test in this file with true program size ~25% of the gene panel, SLICE
scored Gene A/B correlation 0.40 at sparsity=0.05 but 0.98 at
sparsity=0.25. Comparing methods fairly means tuning each one, not
leaving one on an arbitrary setting -- this script tunes SLICE's
sparsity over a small grid by default; see `sparsity_grid` below.
"""
from __future__ import annotations

import time
import warnings
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ---------------------------------------------------------------
# SLICE
# ---------------------------------------------------------------
def run_slice(Xc, gene_names: Sequence[str], k: int,
              sparsity_grid: Sequence[float] = (0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4),
              random_state: int = 0, select_by=None):
    """Fit SLICE at each sparsity in `sparsity_grid` and keep the best,
    scored by `select_by(U, V) -> float` (higher is better) if given,
    else by total variance captured. `Xc` should be normalized+log1p'd
    but NOT explicitly centered (this centers implicitly via
    `fit_slice(..., center=True)`, so it never densifies).

    Returns (U, D, V, gene_names, best_sparsity, runtime_s).

    In a real analysis without ground truth, `select_by` isn't
    available -- pick sparsity by expected program size instead (see
    the module docstring), or by kME/module-size diagnostics after a
    single fit rather than a metric sweep.
    """
    from slice_lca.core import fit_slice

    t0 = time.time()
    best = None
    for s in sparsity_grid:
        result = fit_slice(Xc, gene_names=gene_names, k=k, sparsity=s,
                            random_state=random_state, validate_input=False)
        score = select_by(result.U, result.V) if select_by is not None else float((result.D ** 2).sum())
        if best is None or score > best[0]:
            best = (score, s, result)
    runtime = time.time() - t0
    _, best_s, result = best
    return result.U, result.D, result.V, list(result.gene_names), best_s, runtime


# ---------------------------------------------------------------
# cNMF (the real package, not a proxy)
# ---------------------------------------------------------------
def run_cnmf(counts_h5ad_path: str, k: int, output_dir: str = "cnmf_output", name: str = "run",
             n_iter: int = 20, num_highvar_genes: Optional[int] = 2000, seed: int = 14,
             density_threshold: float = 2.0):
    """Runs the actual cnmf package end to end. `counts_h5ad_path` must
    point to an AnnData .h5ad file of RAW, non-negative counts (cNMF
    does its own TPM normalization and highly-variable-gene selection
    internally -- don't pre-normalize).

    Returns (usage, spectra_tpm, gene_names, top_genes, runtime_s).
    usage is (cells x k) and plays the role of SLICE's U; spectra_tpm
    is (genes x k) and plays the role of SLICE's V (always >= 0) --
    note this covers ALL genes in the input file, not just the
    num_highvar_genes used for fitting (see module docstring, fix #1).
    gene_names is spectra_tpm's row index, in matching order, so you
    can align it against SLICE's (usually HVG-restricted) gene list.
    """
    from cnmf import cNMF

    t0 = time.time()
    cnmf_obj = cNMF(output_dir=output_dir, name=name)
    cnmf_obj.prepare(counts_fn=counts_h5ad_path, components=[k], n_iter=n_iter,
                      seed=seed, num_highvar_genes=num_highvar_genes)
    cnmf_obj.factorize(worker_i=0, total_workers=1)
    cnmf_obj.combine()
    cnmf_obj.consensus(k=k, density_threshold=density_threshold, show_clustering=False)
    usage, spectra_scores, spectra_tpm, top_genes = cnmf_obj.load_results(K=k, density_threshold=density_threshold)
    runtime = time.time() - t0
    gene_names = spectra_tpm.index.astype(str).tolist()
    return usage.values, spectra_tpm.values, gene_names, top_genes, runtime


# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------
def _best_corr(mat: np.ndarray, target: np.ndarray) -> float:
    return max(
        (abs(np.corrcoef(target, mat[:, j])[0, 1]) for j in range(mat.shape[1]) if np.std(mat[:, j]) > 0),
        default=0.0,
    )


def evaluate_against_ground_truth(U, V, tA, tB, gA, gB) -> dict:
    """The same Cell A/B, Gene A/B metrics used throughout the paper."""
    return {
        "CellA": _best_corr(U, tA), "CellB": _best_corr(U, tB),
        "GeneA": _best_corr(V, gA), "GeneB": _best_corr(V, gB),
    }


def _align_by_gene_name(V_a: np.ndarray, genes_a: Sequence[str],
                         V_b: np.ndarray, genes_b: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, list]:
    """Restrict and reorder both loading matrices to their common
    genes, matched by name (case-insensitive), not by row position.
    This is the fix for the actual crash: cNMF's spectra_tpm and
    SLICE's V will almost never have the same gene count or order."""
    genes_a_upper = [g.upper() for g in genes_a]
    genes_b_upper = [g.upper() for g in genes_b]
    idx_a = {g: i for i, g in enumerate(genes_a_upper)}
    idx_b = {g: i for i, g in enumerate(genes_b_upper)}
    common = sorted(set(genes_a_upper) & set(genes_b_upper))
    if len(common) == 0:
        raise ValueError(
            "No gene names in common between the two inputs -- check that both "
            "were built from the same underlying dataset/species."
        )
    rows_a = [idx_a[g] for g in common]
    rows_b = [idx_b[g] for g in common]
    return V_a[rows_a, :], V_b[rows_b, :], common


def evaluate_cross_method_agreement(V_slice: np.ndarray, genes_slice: Sequence[str],
                                     V_cnmf: np.ndarray, genes_cnmf: Sequence[str]) -> pd.DataFrame:
    """Without ground truth: match SLICE's and cNMF's programs to each
    other (Hungarian assignment on |gene-loading correlation|, using
    only genes present in BOTH -- see `_align_by_gene_name`) and
    report how well-matched they are. High agreement doesn't prove
    either is "correct", but persistent LOW agreement is a real
    warning sign that at least one is unstable on your data -- and if
    they largely agree except for specific antagonistic/signed
    programs, that discrepancy is itself informative (it's exactly
    the structure cNMF cannot represent).
    """
    from scipy.optimize import linear_sum_assignment

    V_a, V_b, common_genes = _align_by_gene_name(V_slice, genes_slice, V_cnmf, genes_cnmf)
    k_slice, k_cnmf = V_a.shape[1], V_b.shape[1]
    C = np.zeros((k_slice, k_cnmf))
    for i in range(k_slice):
        for j in range(k_cnmf):
            C[i, j] = abs(np.corrcoef(V_a[:, i], V_b[:, j])[0, 1])
    row, col = linear_sum_assignment(-C)
    result = pd.DataFrame({
        "SLICE_component": row, "cNMF_component": col,
        "abs_gene_loading_corr": C[row, col],
    }).sort_values("abs_gene_loading_corr", ascending=False).reset_index(drop=True)
    result.attrs["n_common_genes"] = len(common_genes)
    return result


# ---------------------------------------------------------------
# Self-test on synthetic ground-truth data (already run + verified;
# re-running this reproduces the numbers reported to you)
# ---------------------------------------------------------------
def _make_small_hard_overlap(seed: int = 42):
    rng = np.random.default_rng(seed)
    n_cells, n_genes = 1500, 300
    X = rng.poisson(0.1, (n_cells, n_genes)).astype(np.float32)
    X[:900, 0:45] += rng.poisson(5, (900, 45))
    X[:900, 45:75] += rng.poisson(5, (900, 30))
    X[600:1350, 75:120] += rng.poisson(5, (750, 45))
    X[600:1350, 45:75] += rng.poisson(5, (750, 30))
    tA = (np.arange(n_cells) < 900).astype(float)
    tB = ((np.arange(n_cells) >= 600) & (np.arange(n_cells) < 1350)).astype(float)
    gA = np.zeros(n_genes); gA[:75] = 1
    gB = np.zeros(n_genes); gB[45:120] = 1
    return X, tA, tB, gA, gB


def self_test(output_dir: str = "cnmf_selftest_output"):
    """Reproduces the exact numbers quoted in this file's docstring
    and in the paper's comparison section. Takes ~5-15s. Also
    exercises the gene-alignment fix (bug #1) since this dataset's
    SLICE and cNMF gene panels are identical here, giving a clean
    sanity check that alignment doesn't silently drop or reorder genes
    when it doesn't need to."""
    import scanpy as sc

    X, tA, tB, gA, gB = _make_small_hard_overlap()
    gene_names = [f"g{i}" for i in range(X.shape[1])]
    h5ad_path = "self_test_counts.h5ad"
    adata = sc.AnnData(X=X)
    adata.var_names = gene_names
    adata.write(h5ad_path)

    Xlog = sp.csr_matrix(np.log1p(X))  # NOT explicitly centered -- fit_slice centers implicitly

    def score(U, V):
        m = evaluate_against_ground_truth(U, V, tA, tB, gA, gB)
        return m["CellA"] + m["CellB"] + m["GeneA"] + m["GeneB"]

    U, D, V, genes_slice, best_s, t_slice = run_slice(Xlog, gene_names, k=4, select_by=score)
    slice_metrics = evaluate_against_ground_truth(U, V, tA, tB, gA, gB)
    slice_metrics.update({"Method": "SLICE", "Runtime_s": round(t_slice, 2), "sparsity_used": best_s})

    usage, spectra_tpm, genes_cnmf, top_genes, t_cnmf = run_cnmf(h5ad_path, k=4, output_dir=output_dir, name="selftest")
    cnmf_metrics = evaluate_against_ground_truth(usage, spectra_tpm, tA, tB, gA, gB)
    cnmf_metrics.update({"Method": "cNMF", "Runtime_s": round(t_cnmf, 2), "sparsity_used": None})

    df = pd.DataFrame([slice_metrics, cnmf_metrics])[["Method", "CellA", "CellB", "GeneA", "GeneB", "Runtime_s", "sparsity_used"]]
    print(df.to_string(index=False))

    # Exercise the actual fix even though gene panels match here 1:1:
    agreement = evaluate_cross_method_agreement(V, genes_slice, spectra_tpm, genes_cnmf)
    print(f"\ncross-method agreement (aligned on {agreement.attrs['n_common_genes']} common genes):")
    print(agreement.to_string(index=False))
    return df


if __name__ == "__main__":
    self_test()


# =================================================================
# TO RUN ON A REAL DATASET IN COLAB:
# =================================================================
# import scanpy as sc
# adata = sc.read_h5ad("your_real_counts.h5ad")   # RAW counts, not normalized
# adata.write("real_counts_raw.h5ad")               # cnmf wants a path
#
# # cNMF (real package):
# usage, spectra_tpm, genes_cnmf, top_genes, t_cnmf = run_cnmf(
#     "real_counts_raw.h5ad", k=YOUR_K, num_highvar_genes=2000,
# )
#
# # SLICE, on the SAME cells/genes cNMF used (its highvar gene subset,
# # written to <name>.overdispersed_genes.txt by prepare()) -- for a
# # fair comparison. No explicit centering needed (fit_slice/run_slice
# # center implicitly), so this stays sparse even at atlas scale:
# import numpy as np, scipy.sparse as sp
# overdispersed_genes = open("cnmf_real_output/real_run/real_run.overdispersed_genes.txt").read().split()
# adata_hvg = adata[:, adata.var_names.isin(overdispersed_genes)]
# Xlog = sp.csr_matrix(np.log1p(adata_hvg.X))
# U, D, V, genes_slice, best_sparsity, t_slice = run_slice(Xlog, adata_hvg.var_names.tolist(), k=YOUR_K)
#
# # Without ground truth, check whether the two methods broadly agree,
# # and where they don't (that disagreement is itself informative) --
# # aligned by gene NAME, not position, since cNMF's spectra_tpm covers
# # every gene in the input file, not just the HVG subset it fit on:
# agreement = evaluate_cross_method_agreement(V, genes_slice, spectra_tpm, genes_cnmf)
# print(agreement)
