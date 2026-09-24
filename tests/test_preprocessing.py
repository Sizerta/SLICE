"""
Tests for slice_lca.preprocessing -- the QC/normalize/HVG/scale chain.
Until this file, this module had no dedicated test coverage at all
(test_centering.py tests the *centering* mechanism in core.py, a
different thing). force_include is tested thoroughly below because
it exists specifically to fix a real failure: variance-based HVG
selection dropped Dclk1/Trpm5 (Tuft cell markers) from a real
analysis entirely -- see select_hvg's docstring.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from slice_lca.preprocessing import (
    qc_filter,
    normalize_log1p,
    select_hvg,
    scale_only,
    preprocess,
)


def _toy_counts(seed=0, n_cells=200, n_genes=50):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.5, (n_cells, n_genes)).astype(np.float32)
    gene_names = [f"gene{i}" for i in range(n_genes)]
    return sp.csr_matrix(X), np.asarray(gene_names, dtype=object)


# ---------------------------------------------------------------------
# qc_filter
# ---------------------------------------------------------------------
def test_qc_filter_removes_all_zero_gene():
    X, genes = _toy_counts()
    X = X.tolil()
    X[:, 0] = 0  # gene0 expressed in zero cells
    X = X.tocsr()
    X_filtered, genes_filtered = qc_filter(X, genes, min_cells_per_gene=1, min_genes_per_cell=0)
    assert "gene0" not in genes_filtered
    assert X_filtered.shape[1] == len(genes_filtered)


def test_qc_filter_removes_low_count_cell():
    X, genes = _toy_counts()
    X = X.tolil()
    X[0, :] = 0  # cell0 has zero genes detected
    X = X.tocsr()
    X_filtered, genes_filtered = qc_filter(X, genes, min_cells_per_gene=0, min_genes_per_cell=1)
    assert X_filtered.shape[0] == X.shape[0] - 1


# ---------------------------------------------------------------------
# normalize_log1p
# ---------------------------------------------------------------------
def test_normalize_log1p_matches_manual_calculation():
    X, genes = _toy_counts(n_cells=20, n_genes=10)
    X_norm = normalize_log1p(X, target_sum=1e4)
    X_dense = np.asarray(X.todense())
    row_sums = X_dense.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    expected = np.log1p(X_dense / row_sums * 1e4)
    np.testing.assert_allclose(np.asarray(X_norm.todense()), expected, rtol=1e-4, atol=1e-5)


def test_normalize_log1p_never_densifies_type():
    X, genes = _toy_counts()
    X_norm = normalize_log1p(X)
    assert sp.issparse(X_norm)


# ---------------------------------------------------------------------
# select_hvg -- basic top-N behavior
# ---------------------------------------------------------------------
def test_select_hvg_keeps_highest_variance_genes():
    rng = np.random.default_rng(0)
    n_cells, n_genes = 300, 50
    X = rng.poisson(0.5, (n_cells, n_genes)).astype(np.float32)
    # gene0 has huge variance; everything else is low-variance noise
    X[:, 0] = rng.poisson(20, n_cells)
    genes = np.asarray([f"gene{i}" for i in range(n_genes)], dtype=object)
    X_reduced, genes_reduced = select_hvg(sp.csr_matrix(X), genes, n_top=5)
    assert "gene0" in genes_reduced
    assert X_reduced.shape[1] == 5


def test_select_hvg_n_top_none_keeps_everything():
    X, genes = _toy_counts()
    X_reduced, genes_reduced = select_hvg(X, genes, n_top=None)
    assert X_reduced.shape[1] == X.shape[1]
    assert list(genes_reduced) == list(genes)


# ---------------------------------------------------------------------
# select_hvg -- force_include (new; this is the fix for the real
# Dclk1/Trpm5 failure)
# ---------------------------------------------------------------------
def test_force_include_keeps_low_variance_marker():
    rng = np.random.default_rng(0)
    n_cells, n_genes = 300, 50
    X = rng.poisson(0.5, (n_cells, n_genes)).astype(np.float32)
    X[:, 0] = rng.poisson(20, n_cells)  # gene0: high variance, would be selected anyway
    X[:, -1] = rng.poisson(0.1, n_cells)  # last gene: deliberately near-zero variance
    genes = np.asarray([f"gene{i}" for i in range(n_genes)], dtype=object)
    marker = genes[-1]

    # without force_include, the low-variance marker is dropped
    _, genes_without = select_hvg(sp.csr_matrix(X), genes, n_top=5)
    assert marker not in genes_without

    # with force_include, it survives even though it wouldn't rank in the top 5
    _, genes_with = select_hvg(sp.csr_matrix(X), genes, n_top=5, force_include=[marker])
    assert marker in genes_with


def test_force_include_is_case_insensitive():
    X, genes = _toy_counts(n_genes=30)
    target = genes[-1]  # e.g. "gene29"
    _, genes_reduced = select_hvg(X, genes, n_top=5, force_include=[target.upper()])
    assert target in genes_reduced


def test_force_include_silently_ignores_absent_genes():
    X, genes = _toy_counts(n_genes=30)
    # should not raise, even though "NOT_A_REAL_GENE" isn't in the panel
    X_reduced, genes_reduced = select_hvg(X, genes, n_top=5, force_include=["NOT_A_REAL_GENE"])
    assert X_reduced.shape[1] == 5


def test_force_include_none_matches_no_force_include_exactly():
    X, genes = _toy_counts(n_genes=30)
    X_a, genes_a = select_hvg(X, genes, n_top=5, force_include=None)
    X_b, genes_b = select_hvg(X, genes, n_top=5)
    assert list(genes_a) == list(genes_b)
    np.testing.assert_array_equal(np.asarray(X_a.todense()), np.asarray(X_b.todense()))


def test_force_include_no_duplicate_when_gene_already_in_top_n():
    rng = np.random.default_rng(0)
    n_cells, n_genes = 300, 50
    X = rng.poisson(0.5, (n_cells, n_genes)).astype(np.float32)
    X[:, 0] = rng.poisson(20, n_cells)  # gene0 will be in the top-N on its own
    genes = np.asarray([f"gene{i}" for i in range(n_genes)], dtype=object)
    X_reduced, genes_reduced = select_hvg(sp.csr_matrix(X), genes, n_top=5, force_include=["gene0"])
    # no duplicate column for gene0
    assert list(genes_reduced).count("gene0") == 1
    assert X_reduced.shape[1] == 5


def test_force_include_gene_names_not_truncated():
    """Regression test for a real bug: building the gene_names array
    with a short dtype (inferred from the input) then assigning a
    longer forced-include name back into it silently truncates the
    string. Use a name longer than any name already in the panel."""
    X, genes = _toy_counts(n_genes=10)  # all names are 5-6 chars ("gene0".."gene9")
    long_name = "a_much_longer_gene_symbol_than_the_rest"
    genes_with_long = genes.copy()
    genes_with_long[-1] = long_name
    _, genes_reduced = select_hvg(X, genes_with_long, n_top=3, force_include=[long_name])
    assert long_name in genes_reduced
    assert genes_reduced[list(genes_reduced).index(long_name)] == long_name  # not truncated


# ---------------------------------------------------------------------
# scale_only
# ---------------------------------------------------------------------
def test_scale_only_unit_variance_per_gene():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 5, (500, 10)).astype(np.float32)
    X_scaled = scale_only(sp.csr_matrix(X))
    X_dense = np.asarray(X_scaled.todense())
    variances = X_dense.var(axis=0)
    np.testing.assert_allclose(variances, np.ones(10), atol=0.05)


# ---------------------------------------------------------------------
# preprocess -- end to end, including force_include passthrough
# ---------------------------------------------------------------------
def test_preprocess_end_to_end_runs_and_returns_matching_shapes():
    # min_genes_per_cell's real default (200) assumes real-dataset scale;
    # this toy panel only has 100 genes total, so it's set to 0 here to
    # isolate what this test actually checks (HVG count, sparsity).
    X, genes = _toy_counts(n_cells=300, n_genes=100)
    X_ready, genes_ready = preprocess(X, genes, n_top_hvg=20, min_genes_per_cell=0)
    assert X_ready.shape[1] == len(genes_ready)
    assert X_ready.shape[1] == 20
    assert sp.issparse(X_ready)


def test_preprocess_force_include_survives_full_pipeline():
    rng = np.random.default_rng(0)
    n_cells, n_genes = 300, 100
    X = rng.poisson(0.5, (n_cells, n_genes)).astype(np.float32)
    X[:, -1] = rng.poisson(0.05, n_cells)  # last gene: very low count/variance
    genes = np.asarray([f"gene{i}" for i in range(n_genes)], dtype=object)
    marker = genes[-1]

    _, genes_without = preprocess(X, genes, n_top_hvg=10, min_cells_per_gene=0, min_genes_per_cell=0)
    assert marker not in genes_without

    _, genes_with = preprocess(X, genes, n_top_hvg=10, min_cells_per_gene=0, min_genes_per_cell=0,
                                 force_include=[marker])
    assert marker in genes_with
