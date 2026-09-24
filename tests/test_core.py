import numpy as np
import pytest
import scipy.sparse as sp

from slice_lca.core import (
    _project_sparse_unit,
    sparsity_budget,
    sparse_svd,
    canonicalize_signs,
    compute_kME,
    assign_modules,
)


def test_sparsity_budget_bounds():
    # s=1.0 -> no constraint (c = sqrt(p))
    assert sparsity_budget(1.0, 1000) == pytest.approx(np.sqrt(1000))
    # s -> 0 -> most sparse possible (c -> 1)
    assert sparsity_budget(0.0, 1000) == 1.0
    with pytest.raises(ValueError):
        sparsity_budget(1.5, 1000)


def test_project_sparse_unit_is_unit_norm():
    rng = np.random.default_rng(0)
    delta = rng.standard_normal(200)
    v = _project_sparse_unit(delta, c=5.0)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)


def test_project_sparse_unit_respects_l1_budget():
    rng = np.random.default_rng(0)
    delta = rng.standard_normal(200)
    c = 5.0
    v = _project_sparse_unit(delta, c=c)
    ratio = np.abs(v).sum() / np.linalg.norm(v)
    assert ratio <= c + 1e-6


def test_project_sparse_unit_no_constraint_when_c_covers_full_vector():
    rng = np.random.default_rng(0)
    delta = rng.standard_normal(50)
    v = _project_sparse_unit(delta, c=np.sqrt(50))
    np.testing.assert_allclose(v, delta / np.linalg.norm(delta), atol=1e-10)


def test_high_sparsity_gives_near_one_hot_component():
    """Regression test for the ablation-table bug: sparsity=0.0 (c=1)
    must yield ~1 nonzero gene per component, NOT a dense solution."""
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.standard_normal((200, 100)))
    _, _, V = sparse_svd(X, k=2, sparsity=0.0, random_state=0)
    nnz_per_component = (np.abs(V) > 1e-8).sum(axis=0)
    assert (nnz_per_component <= 2).all()


def test_full_sparsity_gives_dense_component():
    """sparsity=1.0 must yield fully dense loadings (the true 'no
    sparsity constraint' setting)."""
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.standard_normal((200, 100)))
    _, _, V = sparse_svd(X, k=2, sparsity=1.0, random_state=0)
    nnz_per_component = (np.abs(V) > 1e-8).sum(axis=0)
    assert (nnz_per_component == 100).all()


def test_canonicalize_signs_top_loading_is_positive():
    rng = np.random.default_rng(0)
    V = rng.standard_normal((50, 3))
    U = rng.standard_normal((20, 3))
    _, V_c = canonicalize_signs(U, V)
    top = np.argmax(np.abs(V_c), axis=0)
    assert (V_c[top, np.arange(3)] >= 0).all()


def test_compute_kme_matches_dense_correlation():
    rng = np.random.default_rng(0)
    X_dense = rng.standard_normal((100, 30))
    U = rng.standard_normal((100, 2))
    kME_sparse = compute_kME(sp.csr_matrix(X_dense), U)
    expected = np.column_stack([
        [np.corrcoef(X_dense[:, j], U[:, m])[0, 1] for m in range(2)]
        for j in range(30)
    ]).T
    np.testing.assert_allclose(kME_sparse, expected, atol=1e-6)


def test_assign_modules_respects_threshold():
    kME = np.array([[0.9, 0.1], [0.2, 0.05], [0.5, 0.4]])
    labels, best_val = assign_modules(kME, min_kME=0.3)
    np.testing.assert_array_equal(labels, [0, -1, 0])


def test_sparse_svd_deflate_flag_changes_result():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.standard_normal((300, 150)))
    _, _, V_on = sparse_svd(X, k=3, sparsity=0.1, deflate=True, random_state=0)
    _, _, V_off = sparse_svd(X, k=3, sparsity=0.1, deflate=False, random_state=0)
    assert not np.allclose(V_on, V_off)
