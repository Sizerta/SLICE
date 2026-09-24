import numpy as np
import pytest
import scipy.sparse as sp

from slice_lca.core import fit_slice, NotPreprocessedError, SLICEResult
from slice_lca.datasets import make_antagonistic_axis
from slice_lca.preprocessing import normalize_log1p, scale_only


def _preprocessed_antagonistic_axis(seed=0):
    X, S_true, W_true = make_antagonistic_axis(n_cells=1000, n_genes=500, snr=5.0, seed=seed)
    X_dense = X.toarray()
    X_counts = np.maximum(X_dense - X_dense.min(), 0) + 0.1
    Xs = sp.csr_matrix(X_counts.astype(np.float32))
    X_ready = scale_only(normalize_log1p(Xs))
    gene_names = [f"gene_{i}" for i in range(X_ready.shape[1])]
    return X_ready, gene_names, W_true


def test_fit_slice_rejects_raw_counts_by_default():
    rng = np.random.default_rng(0)
    X_raw = sp.csr_matrix(rng.poisson(3, size=(200, 100)).astype(np.float32))
    with pytest.raises(NotPreprocessedError):
        fit_slice(X_raw, k=3)


def test_fit_slice_validate_input_false_allows_raw_counts_through():
    rng = np.random.default_rng(0)
    X_raw = sp.csr_matrix(rng.poisson(3, size=(200, 100)).astype(np.float32))
    result = fit_slice(X_raw, k=3, validate_input=False)
    assert isinstance(result, SLICEResult)


def test_fit_slice_accepts_properly_preprocessed_data():
    X_ready, gene_names, _ = _preprocessed_antagonistic_axis()
    result = fit_slice(X_ready, gene_names, k=1, sparsity=0.15)
    assert isinstance(result, SLICEResult)
    assert result.V.shape == (500, 1)


def test_fit_slice_center_default_recovers_antagonistic_axis():
    """End-to-end: fit_slice's default settings (center=True) on
    properly preprocessed (but not explicitly centered) input should
    recover a known signed axis well."""
    X_ready, gene_names, W_true = _preprocessed_antagonistic_axis()
    result = fit_slice(X_ready, gene_names, k=1, sparsity=0.15)
    corr = abs(np.corrcoef(W_true[:, 0], result.V[:, 0])[0, 1])
    assert corr > 0.85


def test_fit_slice_center_false_reproduces_old_broken_behavior():
    """Documents the contrast: with center=False (the pre-fix
    default), the same input fails to recover the axis."""
    X_ready, gene_names, W_true = _preprocessed_antagonistic_axis()
    result = fit_slice(X_ready, gene_names, k=1, sparsity=0.15, center=False, validate_input=False)
    corr = abs(np.corrcoef(W_true[:, 0], result.V[:, 0])[0, 1])
    assert corr < 0.3


def test_fit_slice_gene_names_default_and_explicit():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.standard_normal((100, 20)).astype(np.float32))
    r1 = fit_slice(X, k=2, validate_input=False)
    assert r1.gene_names == [f"gene_{i}" for i in range(20)]
    r2 = fit_slice(X, gene_names=[f"g{i}" for i in range(20)], k=2, validate_input=False)
    assert r2.gene_names[0] == "g0"
