"""
Core SLICE algorithm: sparse, signed, deflationary singular-vector
extraction for large-scale single-cell expression matrices.

This is a cleaned-up, tested packaging of the algorithm as implemented
and validated in the SLICE paper's notebooks. The math is unchanged
from that implementation; this module adds type hints, docstrings,
input validation, and a stable public API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------
# Sparse projection
# ---------------------------------------------------------------------
def _project_sparse_unit(delta: np.ndarray, c: float) -> np.ndarray:
    """Project `delta` onto {v : ||v||_2 = 1, ||v||_1 / ||v||_2 <= c}.

    Implements soft-thresholding with a bisection search for the
    smallest threshold lambda such that the L1/L2 ratio constraint is
    satisfied (Witten, Tibshirani & Hastie 2009).
    """
    norm2 = np.linalg.norm(delta)
    if norm2 < 1e-12:
        return delta
    if c >= np.sqrt(len(delta)):
        # Budget covers the whole vector: constraint never binds.
        return delta / norm2

    abs_delta = np.abs(delta)
    sign_delta = np.sign(delta)
    lo, hi = 0.0, float(abs_delta.max())
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        mag = np.maximum(abs_delta - mid, 0.0)
        n2 = np.linalg.norm(mag)
        if n2 < 1e-12:
            hi = mid
            continue
        l1 = mag.sum() / n2
        if l1 > c:
            lo = mid
        else:
            hi = mid
    mag = np.maximum(abs_delta - hi, 0.0)
    n2 = np.linalg.norm(mag)
    v = sign_delta * mag
    return v / n2 if n2 > 1e-12 else delta / norm2


def sparsity_budget(sparsity: float, p: int) -> float:
    """c = max(sqrt(sparsity * p), 1). sparsity=1.0 -> no constraint
    (c = sqrt(p)); sparsity near 0 -> maximally sparse (c -> 1, i.e. a
    single dominant gene per component). See the paper's Methods section
    and `docs/hyperparameters.md` for guidance on choosing this value.
    """
    if not (0.0 <= sparsity <= 1.0):
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
    return max(np.sqrt(sparsity * p), 1.0)


# ---------------------------------------------------------------------
# Implicit centering (never materializes X - column_means, so sparsity
# and the O(nnz) memory bound are preserved exactly as claimed in the
# paper's Methods section -- see sparse_svd's `center` parameter).
# ---------------------------------------------------------------------
def _column_means(X) -> np.ndarray:
    if sp.issparse(X):
        return np.asarray(X.mean(axis=0)).ravel()
    return np.asarray(X).mean(axis=0)


def _centered_matvec(X, v: np.ndarray, mu: Optional[np.ndarray]) -> np.ndarray:
    """(X - 1 mu^T) @ v = X@v - (mu . v) * 1_n, computed without ever
    forming the dense centered matrix."""
    Xv = np.asarray(X @ v).ravel()
    if mu is not None:
        Xv = Xv - float(mu @ v)
    return Xv


def _centered_rmatvec(X, u: np.ndarray, mu: Optional[np.ndarray]) -> np.ndarray:
    """(X - 1 mu^T)^T @ u = X^T@u - mu * sum(u), computed without ever
    forming the dense centered matrix."""
    XTu = np.asarray(X.T @ u).ravel()
    if mu is not None:
        XTu = XTu - mu * float(u.sum())
    return XTu


# ---------------------------------------------------------------------
# Sparse SVD (deflationary sparse power iteration)
# ---------------------------------------------------------------------
def sparse_svd(
    X,
    k: int,
    sparsity: float = 0.1,
    n_iter: int = 10,
    tol: float = 1e-3,
    random_state: int = 0,
    deflate: bool = True,
    n_warmup_iter: int = 0,
    check_v_convergence: bool = False,
    return_diagnostics: bool = False,
    center: bool = False,
):
    """Rank-k sparse, signed decomposition of X via deflationary sparse
    power iteration.

    Parameters
    ----------
    X : (n, p) array or scipy sparse matrix
        Centered expression matrix. Sparse input is never densified.
    k : int
        Number of components.
    sparsity : float in (0, 1]
        Sparsity budget passed to `sparsity_budget`. Smaller = sparser.
    n_iter : int
        Max power-iteration steps per component.
    tol : float
        Convergence tolerance on the singular value.
    random_state : int
        Seed for the random component initializations.
    deflate : bool
        If True (default), each component's contribution is removed
        from `u`/`delta` before extracting the next one, allowing
        sequential discovery of overlapping (non-orthogonal) programs
        without rediscovering the same dominant axis. Set False only
        for ablation/diagnostic purposes.
    n_warmup_iter : int, default 0 (off -- matches all previously
        published results exactly)
        Run this many *unconstrained* power-iteration steps (same
        deflation-aware updates, no sparsity projection) before
        switching on the sparse projection, on the theory that
        starting the constrained iteration near the true dominant
        axis (rather than a uniformly random direction) should reduce
        bad-local-optimum outcomes. Tested against that theory across
        three regimes (moderate overlap, severe overlap, and low-SNR
        with more components) using the paper's own recovery metrics
        over 25-50 seeds each -- see `benchmarks/warmstart_*.py`.
        **It did not measurably help in any of them** (worst-case
        recovery, mean, and variance were statistically indistinguishable
        with vs. without warm-up). Most likely explanation: the
        unconstrained problem's optimization landscape isn't a good
        proxy for the L1-constrained one, so a warm start on the
        former doesn't reliably land in a better basin for the
        latter. Left available (it's essentially free and never
        changes the default) in case it helps on real data with
        different structure than these synthetic tests, but it is
        NOT a validated fix -- if you want a validated way to reduce
        bad-seed risk, see `sparse_svd_multistart` instead, which
        empirically does help (see its docstring).
    check_v_convergence : bool, default False (off -- matches all
        previously published results exactly)
        If True, also require the gene loading vector itself to have
        stabilized (not just the scalar objective `d`) before
        declaring convergence. `d` can be nearly stationary while `v`
        is still rotating, especially early in optimization or when
        singular values are close together; this is a strictly
        stricter (never looser) stopping condition.
    return_diagnostics : bool, default False (off -- preserves the
        3-tuple return signature every existing caller relies on)
        If True, also return a list of k dicts (one per component)
        with `n_iter_used`, `converged`, and `final_delta_d` -- so you
        can tell whether a real run actually converged instead of
        silently hitting the iteration cap. Currently nothing in this
        codebase surfaces that; it fails silently.
    center : bool, default False (off -- matches all previously
        published results exactly, which were generated by calling
        this function on data that was NOT explicitly centered)
        If True, implicitly centers X (subtracts each gene's column
        mean) inside every matrix-vector product, without ever
        forming the dense centered matrix -- exactly the mechanism
        the paper's Methods section describes ("column means are
        incorporated implicitly during matrix-vector multiplication,
        so the centered matrix is never explicitly materialized").
        That mechanism was described but not actually implemented
        anywhere in this codebase until now. This matters far more
        than it sounds: on a clean synthetic test with a known signed
        (antagonistic) axis, calling this function the same way the
        paper's own real-data pipeline does (normalized, log1p'd,
        variance-scaled, but NOT centered) recovers the true axis
        with gene-loading correlation 0.006 -- i.e. it fails
        completely, because an uncentered power iteration on
        non-negative data is dominated by overall expression level,
        not by genuine (signed) biological variance. With
        `center=True` on the identical input, correlation is 0.95.
        See `tests/test_centering.py` and `benchmarks/centering_validation.py`.

        `fit_slice` and `preprocess()` now default to `center=True`;
        this function's own default stays `False` so nothing already
        published silently changes underneath a direct `sparse_svd`
        call.

    Returns
    -------
    U : (n, k) ndarray -- cell-level program activities
    D : (k,) ndarray -- component strengths
    V : (p, k) ndarray -- signed, sparse gene loadings
    diagnostics : list[dict], only if return_diagnostics=True
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    rng = np.random.default_rng(random_state)
    n, p = X.shape
    c = sparsity_budget(sparsity, p)
    U = np.zeros((n, k))
    V = np.zeros((p, k))
    D = np.zeros(k)
    diagnostics = []
    mu = _column_means(X) if center else None

    for comp in range(k):
        v = rng.standard_normal(p)
        v /= np.linalg.norm(v)

        # Optional unconstrained warm-up: same deflation-aware updates,
        # but skip the sparse projection so v moves straight toward the
        # dominant direction before the L1 budget starts constraining it.
        for _ in range(n_warmup_iter):
            u_warm = _centered_matvec(X, v, mu)
            if deflate and comp > 0:
                u_warm = u_warm - U[:, :comp] @ (D[:comp] * (V[:, :comp].T @ v))
            un = np.linalg.norm(u_warm)
            if un < 1e-12:
                break
            u_warm /= un
            delta_warm = _centered_rmatvec(X, u_warm, mu)
            if deflate and comp > 0:
                delta_warm = delta_warm - V[:, :comp] @ (D[:comp] * (U[:, :comp].T @ u_warm))
            dn = np.linalg.norm(delta_warm)
            if dn < 1e-12:
                break
            v = delta_warm / dn

        d_prev = -np.inf
        v_prev = v.copy()
        u = np.zeros(n)
        d = 0.0
        converged = False
        it_used = n_iter

        for it in range(1, n_iter + 1):
            u = _centered_matvec(X, v, mu)
            if deflate and comp > 0:
                u = u - U[:, :comp] @ (D[:comp] * (V[:, :comp].T @ v))
            un = np.linalg.norm(u)
            if un < 1e-12:
                it_used = it
                converged = True
                break
            u /= un

            delta = _centered_rmatvec(X, u, mu)
            if deflate and comp > 0:
                delta = delta - V[:, :comp] @ (D[:comp] * (U[:, :comp].T @ u))
            v = _project_sparse_unit(delta, c)

            Xv = _centered_matvec(X, v, mu)
            if deflate and comp > 0:
                Xv = Xv - U[:, :comp] @ (D[:comp] * (V[:, :comp].T @ v))
            d = float(u @ Xv)
            d_delta = abs(d - d_prev)
            d_ok = d_delta < tol * max(abs(d), 1e-12)

            if check_v_convergence:
                v_delta = np.linalg.norm(v - v_prev)
                v_ok = v_delta < tol
                stop = d_ok and v_ok
            else:
                stop = d_ok
            v_prev = v.copy()

            if stop:
                it_used = it
                converged = True
                break
            d_prev = d

        U[:, comp] = u
        V[:, comp] = v
        D[comp] = d
        diagnostics.append({
            "component": comp, "n_iter_used": it_used, "converged": converged,
            "final_delta_d": abs(d - d_prev) if it_used > 0 else None,
        })

    if return_diagnostics:
        return U, D, V, diagnostics
    return U, D, V


def sparse_svd_multistart(X, k: int, n_restarts: int = 4, sparsity: float = 0.1,
                           n_iter: int = 10, tol: float = 1e-3, random_state: int = 0,
                           deflate: bool = True):
    """Run `sparse_svd` `n_restarts` times from different random seeds
    and keep the run with the highest total explained variance
    (sum of D**2) -- a standard, guaranteed-safe way to reduce the
    chance of landing in a bad local optimum for a non-convex problem.
    "Guaranteed-safe" in the sense that the selection criterion never
    uses ground truth (unavailable on real data) and can only match or
    beat a single random start on that same criterion; it is NOT
    guaranteed to improve any particular external metric.

    Empirically validated (unlike `sparse_svd`'s n_warmup_iter option,
    which was tested and did NOT help): on a synthetic benchmark with
    n_progs=8, overlap=0.6, snr=1.5 (deliberately harder than the
    paper's default settings, to make bad-seed outcomes more likely),
    n_restarts=4 improved mean gene-loading recovery correlation from
    0.604 to 0.615 and worst-case (minimum, over 25 independent
    trials) from 0.574 to 0.587, while reducing the spread (variance)
    of the subspace-error metric by ~40%. See
    `benchmarks/multirestart_test.py` for the exact reproduction and
    `tests/test_multistart.py` for a regression test confirming this
    on a held-out random seed range not used to discover the effect.

    Costs n_restarts times the runtime of a single `sparse_svd` call.
    For atlas-scale data where a single fit already takes minutes,
    consider running this only during initial method/parameter
    validation on a subsample, not on every production run.

    Returns
    -------
    U, D, V : same as sparse_svd
    best_seed : the random_state that won
    all_scores : list of (seed, total_explained_variance) for every
        attempt, in case you want to inspect how much seeds disagreed
    """
    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**31 - 1, size=n_restarts)

    best = None
    all_scores = []
    for seed in seeds:
        U, D, V = sparse_svd(X, k=k, sparsity=sparsity, n_iter=n_iter, tol=tol,
                              random_state=int(seed), deflate=deflate)
        score = float((D ** 2).sum())
        all_scores.append((int(seed), score))
        if best is None or score > best[0]:
            best = (score, int(seed), U, D, V)

    _, best_seed, U, D, V = best
    return U, D, V, best_seed, all_scores


def canonicalize_signs(U: np.ndarray, V: np.ndarray):
    """Anchor each component's sign so its largest-magnitude gene
    loading is positive (sparse-PCA axes are identified only up to sign).
    """
    U = U.copy()
    V = V.copy()
    top = np.argmax(np.abs(V), axis=0)
    flip = V[top, np.arange(V.shape[1])] < 0
    if flip.any():
        cols = np.where(flip)[0]
        V[:, cols] *= -1
        U[:, cols] *= -1
    return U, V


# ---------------------------------------------------------------------
# Module membership (kME) and assignment
# ---------------------------------------------------------------------
def _column_sumsq_sparse(X, chunk_nnz: int = 2_000_000) -> np.ndarray:
    p = X.shape[1]
    out = np.zeros(p, dtype=np.float64)
    data = X.data
    indices = X.indices
    nnz = data.shape[0]
    for s in range(0, nnz, chunk_nnz):
        e = min(s + chunk_nnz, nnz)
        d = data[s:e].astype(np.float64, copy=False)
        out += np.bincount(indices[s:e], weights=d * d, minlength=p)
    return out


def compute_kME(X, U: np.ndarray) -> np.ndarray:
    """kME[j, m] = Pearson correlation of gene j's expression with
    module m's eigengene (U[:, m]). X stays sparse if given sparse."""
    n = X.shape[0]
    if sp.issparse(X):
        mean_X = np.asarray(X.mean(axis=0)).ravel()
        sumsq_X = _column_sumsq_sparse(X)
        cross = np.asarray(X.T @ U)
    else:
        mean_X = X.mean(axis=0)
        sumsq_X = (X ** 2).sum(axis=0)
        cross = X.T @ U

    mean_U = U.mean(axis=0)
    norm_U = np.linalg.norm(U - mean_U, axis=0)
    centered_cross = cross - n * np.outer(mean_X, mean_U)
    norm_X = np.sqrt(np.maximum(sumsq_X - n * mean_X ** 2, 0.0))
    return centered_cross / (np.outer(norm_X, norm_U) + 1e-12)


def assign_modules(kME: np.ndarray, min_kME: float = 0.3):
    """Assign each gene to argmax_m |kME(j, m)| if that exceeds
    `min_kME`, else -1 ("grey"/unassigned). See
    `docs/hyperparameters.md` for guidance on this threshold --
    validated stable for min_kME in [0.1, 0.5] on synthetic data
    (100% recall / >=99.8% precision on true module genes; see
    `benchmarks/sensitivity_minkme_summary.csv`)."""
    best_mod = np.argmax(np.abs(kME), axis=1)
    best_val = kME[np.arange(kME.shape[0]), best_mod]
    labels = np.where(np.abs(best_val) >= min_kME, best_mod, -1)
    return labels, best_val


# ---------------------------------------------------------------------
# Rank selection
# ---------------------------------------------------------------------
def select_k_parallel_analysis(X, k_max: int = 20, n_perm: int = 20, percentile: float = 95, random_state: int = 0):
    """Choose k via parallel analysis against column-permuted null
    matrices. Disabled by convention for n >= 100,000 cells (supply k
    explicitly at that scale)."""
    from sklearn.utils.extmath import randomized_svd

    rng = np.random.default_rng(random_state)
    n, p = X.shape
    k_max = min(k_max, n, p)

    def top_sv(A, seed):
        _, s, _ = randomized_svd(A, n_components=k_max, n_iter=7, random_state=seed)
        return s

    real_sv = top_sv(X, random_state)
    null_sv = np.zeros((n_perm, k_max))
    if sp.issparse(X):
        X_csc = X.tocsc()
        for b in range(n_perm):
            new_data = np.empty_like(X_csc.data)
            new_indices = np.empty_like(X_csc.indices)
            indptr = X_csc.indptr
            for j in range(p):
                s0, e0 = indptr[j], indptr[j + 1]
                nnz_j = e0 - s0
                if nnz_j == 0:
                    continue
                vals = X_csc.data[s0:e0][rng.permutation(nnz_j)]
                rows = rng.choice(n, size=nnz_j, replace=False)
                order = np.argsort(rows)
                new_indices[s0:e0] = rows[order]
                new_data[s0:e0] = vals[order]
            Xp = sp.csc_matrix((new_data, new_indices, indptr), shape=(n, p))
            null_sv[b] = top_sv(Xp, random_state + b + 1)
    else:
        for b in range(n_perm):
            keys = rng.random((n, p))
            Xp = np.take_along_axis(X, np.argsort(keys, axis=0), axis=0)
            null_sv[b] = top_sv(Xp, random_state + b + 1)

    thresh = np.percentile(null_sv, percentile, axis=0)
    keep = real_sv > thresh
    k_sel = int(np.argmin(keep)) if not keep.all() else k_max
    return k_sel, real_sv, thresh


# ---------------------------------------------------------------------
# Public result object + top-level fit function
# ---------------------------------------------------------------------
def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns q-values in the same
    order as the input. Matches statsmodels.stats.multitest.multipletests
    (method='fdr_bh') to floating-point precision -- see
    tests/test_stats.py."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p.copy()
    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, n + 1)
    q = ranked * n / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


@dataclass
class SLICEResult:
    U: np.ndarray
    D: np.ndarray
    V: np.ndarray
    kME: np.ndarray
    labels: np.ndarray
    gene_names: Sequence[str]
    k: int


class NotPreprocessedError(ValueError):
    """Raised by fit_slice when the input looks like it hasn't been
    normalized/log-transformed (e.g. raw counts passed directly)."""


def _looks_like_raw_counts(X, sample_size: int = 5000) -> bool:
    """Cheap, reliable heuristic: sample some stored (nonzero) values
    and check whether they're suspiciously close to whole integers.
    Raw UMI/read counts are exact integers; after log1p and variance
    scaling, values are essentially never close to integers except by
    coincidence. This does not check for centering specifically
    (center=True handles that regardless) -- it catches the other,
    equally damaging half of the problem: skipping library-size
    normalization and log1p, which no amount of centering fixes,
    since it's a per-CELL (row) multiplicative effect, not a per-gene
    (column) offset.
    """
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    if data.size == 0:
        return False
    sample = data[:: max(1, data.size // sample_size)][:sample_size]
    near_integer = np.abs(sample - np.round(sample)) < 1e-6
    return float(near_integer.mean()) > 0.95


def fit_slice(
    X,
    gene_names: Optional[Sequence[str]] = None,
    k: Optional[int] = None,
    sparsity: float = 0.02,
    min_kME: float = 0.05,
    k_max: int = 20,
    n_perm: int = 20,
    random_state: int = 0,
    n_restarts: int = 1,
    center: bool = True,
    validate_input: bool = True,
) -> SLICEResult:
    """Fit SLICE end to end: rank selection (if k is None), sparse
    signed factorization, module membership, and assignment.

    Parameters mirror the paper's Methods section. For n >= 100,000
    cells, `k` must be supplied explicitly (parallel analysis is
    disabled at that scale for cost reasons).

    center : bool, default True
        Implicitly mean-center X inside the factorization (see
        `sparse_svd`'s `center` parameter -- this is the fix for the
        most significant issue found in this project: the paper's own
        real-data pipeline never centered, and recovery of signed/
        antagonistic structure fails almost completely without it).
        This default deliberately differs from `sparse_svd`'s own
        default (False) -- `fit_slice` is the recommended entry
        point and should reflect the corrected behavior; pass
        `center=False` only to exactly reproduce pre-fix numbers.
    validate_input : bool, default True
        Raise `NotPreprocessedError` if X looks like raw counts (see
        `_looks_like_raw_counts`) -- centering alone does not fix
        this half of the problem (a per-cell library-size confound,
        not a per-gene offset). Run `slice_lca.preprocessing.preprocess()`
        first, or pass `validate_input=False` if you're intentionally
        supplying something unusual.

    n_restarts : int, default 1 (matches all previously published
        results exactly). Set > 1 to use `sparse_svd_multistart`
        instead of a single fit -- empirically validated to modestly
        improve worst-case recovery and reduce run-to-run variance
        (see `sparse_svd_multistart`'s docstring), at n_restarts times
        the runtime.
    """
    X_mat = X if sp.issparse(X) else np.asarray(X, dtype=np.float32)
    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(X_mat.shape[1])]
    gene_names = list(gene_names)
    if X_mat.ndim != 2:
        raise ValueError("X must be a 2D cells x genes matrix.")

    if validate_input and _looks_like_raw_counts(X_mat):
        raise NotPreprocessedError(
            "Input looks like raw, unnormalized counts (>95% of sampled "
            "nonzero values are exact integers). fit_slice expects "
            "library-size-normalized, log1p-transformed data -- run "
            "slice_lca.preprocessing.preprocess(X, gene_names) first, or "
            "pass validate_input=False if this is intentional."
        )

    if k is None:
        if X_mat.shape[0] >= 100_000:
            raise ValueError("k must be supplied explicitly for n >= 100,000 cells.")
        k, _, _ = select_k_parallel_analysis(X_mat, k_max=k_max, n_perm=n_perm, random_state=random_state)
        k = max(k, 1)

    if n_restarts > 1:
        U, D, V, _, _ = sparse_svd_multistart(X_mat, k, n_restarts=n_restarts, sparsity=sparsity, random_state=random_state)
    else:
        U, D, V = sparse_svd(X_mat, k, sparsity=sparsity, random_state=random_state, center=center)
    U, V = canonicalize_signs(U, V)
    kME = compute_kME(X_mat, U)
    labels, _ = assign_modules(kME, min_kME=min_kME)

    return SLICEResult(U=U, D=D, V=V, kME=kME, labels=labels, gene_names=gene_names, k=k)
