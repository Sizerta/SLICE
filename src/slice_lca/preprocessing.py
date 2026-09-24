"""
slice_lca.preprocessing
QC filter -> library-size normalize + log1p -> HVG selection ->
per-gene variance scale, all sparse-native (never densifies), matching
the paper's Methods section ("Input representation and preprocessing").

WHY THIS MODULE EXISTS: it didn't, until this file. The algorithm core
(`slice_lca.core`) was packaged, tested, and released -- but the
preprocessing steps needed to get raw counts into the form `fit_slice`
expects only ever existed as ad hoc code pasted into individual
analysis notebooks. That gap is exactly what caused a real failure:
a real-data cell called `fit_slice()` directly on raw, un-normalized
counts (no QC, no log1p, no variance-scaling, no HVG restriction) and
got degenerate "modules" claiming 10,000+ genes each.

That failure has a second, deeper layer, which turned out to matter
far more: even the paper's OWN real-dataset pipeline (`run_lca_pipeline`
-> `fit_lca`, used for every Table 1 number) does QC + normalize+log1p
+ HVG + variance-scale, but never explicitly mean-centers the data --
and `sparse_svd` never centered it implicitly either, despite the
paper's Methods section claiming it does. On a clean synthetic test
with a known signed axis, this combination recovers essentially
nothing (correlation 0.006). `sparse_svd` now has a `center` parameter
that implements real implicit centering (mathematically exact, never
densifies -- see `core.py`), and `fit_slice` uses it by default. This
module's `preprocess()` deliberately stops at variance-scaling and
leaves centering to `fit_slice`, rather than densifying here.
"""
from __future__ import annotations

import gc
from typing import Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


def _as_csr_float32(X, copy: bool = False) -> sp.csr_matrix:
    if sp.issparse(X):
        if not sp.isspmatrix_csr(X):
            X = X.tocsr(copy=copy)
        elif copy:
            X = X.copy()
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=True)
        return X
    return sp.csr_matrix(np.asarray(X, dtype=np.float32))


def qc_filter(X, gene_names: Sequence[str], min_cells_per_gene: int = 3,
              min_genes_per_cell: int = 200) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Drop low-quality cells (fewer than `min_genes_per_cell` detected
    genes -- filters out empty/ambient droplets) and rarely-detected
    genes (fewer than `min_cells_per_gene` cells). This is the step
    that turns e.g. 585,000 raw droplets into the ~34,000 real cells
    reported in the paper's Table 1 -- most rows in a raw, unfiltered
    10x matrix are empty droplets, not cells.
    """
    X = _as_csr_float32(X)
    gene_names = np.asarray(gene_names)
    if X.shape[1] != len(gene_names):
        raise ValueError(f"X has {X.shape[1]} genes but got {len(gene_names)} gene_names.")

    keep_cells = np.diff(X.indptr) >= min_genes_per_cell
    if not np.all(keep_cells):
        X = X[keep_cells].tocsr()
    gc.collect()

    gene_counts = np.bincount(X.indices, minlength=X.shape[1])
    keep_genes = gene_counts >= min_cells_per_gene
    if not np.all(keep_genes):
        X = X[:, keep_genes].tocsr()
        gene_names = gene_names[keep_genes]
    gc.collect()
    return X, gene_names


def normalize_log1p(X, target_sum: float = 10_000.0, row_block: int = 4096) -> sp.csr_matrix:
    """Library-size normalize each cell to `target_sum` total counts,
    then log1p. In-place on a CSR copy; processes in row blocks so
    peak memory stays bounded regardless of matrix size."""
    X = _as_csr_float32(X, copy=True)
    n = X.shape[0]

    for r0 in range(0, n, row_block):
        r1 = min(r0 + row_block, n)
        p0, p1 = int(X.indptr[r0]), int(X.indptr[r1])
        if p1 <= p0:
            continue
        row_nnz = np.diff(X.indptr[r0:r1 + 1])
        lib = np.asarray(X[r0:r1].sum(axis=1)).ravel().astype(np.float32, copy=False)
        lib[lib == 0] = 1.0
        scale = np.float32(target_sum) / lib
        repeated = np.repeat(scale, row_nnz)
        X.data[p0:p1] *= repeated
        np.log1p(X.data[p0:p1], out=X.data[p0:p1])

    return X


def _column_moments(X: sp.csr_matrix, row_block: int = 4096):
    n, p = X.shape
    sum_x = np.zeros(p, dtype=np.float64)
    sum_x2 = np.zeros(p, dtype=np.float64)
    for r0 in range(0, n, row_block):
        r1 = min(r0 + row_block, n)
        p0, p1 = int(X.indptr[r0]), int(X.indptr[r1])
        if p1 <= p0:
            continue
        idx = X.indices[p0:p1]
        data = X.data[p0:p1]
        sq = data.astype(np.float64, copy=True)
        sq *= sq
        sum_x += np.bincount(idx, weights=data, minlength=p)
        sum_x2 += np.bincount(idx, weights=sq, minlength=p)
    mean = sum_x / max(n, 1)
    second_moment = sum_x2 / max(n, 1)
    return mean, second_moment


def select_hvg(X, gene_names: Sequence[str], n_top: Optional[int] = 3000,
               row_block: int = 4096,
               force_include: Optional[Sequence[str]] = None) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Keep the `n_top` highest-variance genes (sparse mean/variance,
    never densifies). `n_top=None` keeps every gene.

    force_include : genes to keep regardless of variance rank, matching
        by symbol, case-insensitively. Silently ignores names not
        present in `gene_names` (does not error -- a caller passing a
        marker-gene list that includes one gene absent from this
        dataset should not lose the whole HVG selection). Real finding
        that motivated this parameter: variance-based HVG selection
        can drop genes that carry the biology you care about even
        though they are lowly expressed -- e.g. Dclk1/Trpm5 (Tuft
        cell markers) were HVG-filtered out entirely in an early real
        analysis, at kME below 0.15 for a marker later confirmed
        published at 0.91.
    """
    X = _as_csr_float32(X)
    gene_names = np.asarray(gene_names, dtype=object)
    if n_top is None or n_top >= X.shape[1]:
        return X, gene_names

    mean, second_moment = _column_moments(X, row_block=row_block)
    var = np.maximum(second_moment - mean * mean, 0.0)

    n_top = int(n_top)
    if n_top <= 0:
        raise ValueError("n_top must be positive or None.")
    idx = np.argpartition(-var, n_top - 1)[:n_top]

    if force_include:
        wanted = {g.upper() for g in force_include}
        name_to_idx = {str(g).upper(): i for i, g in enumerate(gene_names)}
        forced_idx = [name_to_idx[g] for g in wanted if g in name_to_idx]
        if forced_idx:
            idx = np.union1d(idx, np.asarray(forced_idx, dtype=idx.dtype))
    idx.sort()

    X_reduced = X[:, idx].tocsr()
    genes_reduced = gene_names[idx]
    del mean, second_moment, var, idx
    gc.collect()
    return X_reduced, genes_reduced


def scale_only(X, row_block_nnz: int = 2_000_000) -> sp.csr_matrix:
    """Per-gene variance scaling (unit variance). Deliberately does
    NOT subtract column means -- centering is handled implicitly by
    `sparse_svd(..., center=True)` (the default in `fit_slice`),
    which applies the correction inside each matrix-vector product
    without ever densifying X. Do this step here as well and you will
    double-center (the explicit-vs-implicit correction would compose
    incorrectly); use this function for scaling only.
    """
    X = _as_csr_float32(X, copy=True)
    mean, second = _column_moments(X)
    std = np.sqrt(np.maximum(second - mean * mean, 1e-12))
    inv_std = (1.0 / std).astype(np.float32, copy=False)
    for s in range(0, X.nnz, row_block_nnz):
        e = min(s + row_block_nnz, X.nnz)
        X.data[s:e] *= inv_std[X.indices[s:e]]
    del mean, second, std, inv_std
    gc.collect()
    return X


def center_explicit(X) -> sp.csr_matrix:
    """Explicitly mean-center X (subtract each gene's column mean).

    You should rarely need this. It DENSIFIES internally (a centered
    matrix generally has no zeros left to exploit), which is exactly
    what `sparse_svd(..., center=True)` was built to avoid -- prefer
    that for anything at real dataset scale. This function exists for
    interfacing with other tools that require an already-materialized
    centered array (e.g. comparing against a method that has no
    implicit-centering option of its own), and is only advisable when
    the gene count is already restricted to a few thousand (HVG-level),
    not tens of thousands.
    """
    X = _as_csr_float32(X)
    Xd = X.toarray()
    Xd -= Xd.mean(axis=0, keepdims=True)
    return sp.csr_matrix(Xd)


def preprocess(
    X,
    gene_names: Sequence[str],
    n_top_hvg: Optional[int] = 3000,
    min_cells_per_gene: int = 3,
    min_genes_per_cell: int = 200,
    skip_qc: bool = False,
    target_sum: float = 10_000.0,
    force_include: Optional[Sequence[str]] = None,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """QC -> normalize+log1p -> HVG -> variance-scale, in one call.
    This is the standard preprocessing chain the paper's Table 1
    numbers were generated with. Returns (X_ready, gene_names_ready).

    force_include : passed through to `select_hvg` -- genes kept
        regardless of variance rank (case-insensitive; silently
        ignores names absent from this dataset). Use this for marker
        genes or pathway members you already know matter and cannot
        afford to lose to HVG filtering; see `select_hvg`'s docstring
        for the real case that motivated this.

    Deliberately does NOT explicitly center -- pass X_ready straight
    to `fit_slice(X_ready, gene_names_ready, ...)`, which centers
    implicitly by default (`center=True`; see `sparse_svd`'s `center`
    parameter). Calling `center_explicit()` on the result yourself is
    unnecessary and, at real dataset scale, exactly the memory-blowing
    step this whole sparse design exists to avoid.
    """
    if not skip_qc:
        X, gene_names = qc_filter(X, gene_names, min_cells_per_gene, min_genes_per_cell)
    X = normalize_log1p(X, target_sum=target_sum)
    if n_top_hvg is not None and X.shape[1] > n_top_hvg:
        X, gene_names = select_hvg(X, gene_names, n_top=n_top_hvg, force_include=force_include)
    X = scale_only(X)
    return X, gene_names
