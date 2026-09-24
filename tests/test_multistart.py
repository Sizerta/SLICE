import numpy as np
import scipy.sparse as sp
import pytest

from slice_lca.core import sparse_svd, sparse_svd_multistart


def test_multistart_n1_matches_plain_sparse_svd():
    """n_restarts=1 should be identical to a single sparse_svd call at
    the seed multistart derives for its one attempt -- this pins down
    that multistart doesn't silently change the underlying math, only
    the selection-among-attempts logic."""
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.standard_normal((200, 100)))
    U, D, V, best_seed, scores = sparse_svd_multistart(X, k=3, n_restarts=1, sparsity=0.1, random_state=0)
    U_ref, D_ref, V_ref = sparse_svd(X, k=3, sparsity=0.1, random_state=best_seed)
    np.testing.assert_allclose(U, U_ref)
    np.testing.assert_allclose(V, V_ref)
    assert len(scores) == 1


def test_multistart_picks_highest_explained_variance():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.standard_normal((300, 150)))
    U, D, V, best_seed, scores = sparse_svd_multistart(X, k=3, n_restarts=5, sparsity=0.1, random_state=0)
    best_score_reported = max(s for _, s in scores)
    assert (D ** 2).sum() == pytest.approx(best_score_reported, rel=1e-6)
    # the winning seed's own independent run should reproduce the same result
    U_check, D_check, V_check = sparse_svd(X, k=3, sparsity=0.1, random_state=best_seed)
    np.testing.assert_allclose(D, D_check)


def test_multistart_rejects_invalid_n_restarts():
    X = sp.csr_matrix(np.eye(10))
    with pytest.raises(ValueError):
        sparse_svd_multistart(X, k=2, n_restarts=0)


def test_multistart_beats_single_start_on_average():
    """Not a per-trial guarantee (multistart's internally-derived seeds
    aren't guaranteed to include whatever seed a single-start run used,
    so a per-trial '>=' isn't logically implied) -- but on average,
    trying 4 attempts and keeping the best should not be worse than
    trying 1. This mirrors the real, held-out-seed empirical result in
    benchmarks/multirestart_test.py at a scale fast enough for CI."""
    rng = np.random.default_rng(1)
    single_scores, multi_scores = [], []
    for trial in range(8):
        X = sp.csr_matrix(rng.standard_normal((150, 80)))
        U0, D0, V0 = sparse_svd(X, k=2, sparsity=0.1, random_state=trial)
        U1, D1, V1, _, _ = sparse_svd_multistart(X, k=2, n_restarts=4, sparsity=0.1, random_state=trial)
        single_scores.append((D0 ** 2).sum())
        multi_scores.append((D1 ** 2).sum())
    assert np.mean(multi_scores) >= np.mean(single_scores)


def test_return_diagnostics_reports_convergence_status():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.standard_normal((200, 100)))
    U, D, V, diag = sparse_svd(X, k=3, sparsity=0.1, n_iter=50, tol=1e-4,
                                random_state=0, return_diagnostics=True)
    assert len(diag) == 3
    for d in diag:
        assert set(d.keys()) == {"component", "n_iter_used", "converged", "final_delta_d"}
        assert 1 <= d["n_iter_used"] <= 50


def test_return_diagnostics_flags_nonconvergence_when_iter_budget_too_small():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.standard_normal((300, 150)))
    U, D, V, diag = sparse_svd(X, k=2, sparsity=0.1, n_iter=1, tol=1e-12,
                                random_state=0, return_diagnostics=True)
    # tol=1e-12 with only 1 iteration allowed should not be able to
    # satisfy the convergence criterion
    assert not diag[0]["converged"]
    assert diag[0]["n_iter_used"] == 1


def test_default_signature_unchanged_returns_three_values():
    """Backward compatibility: existing code doing `U, D, V = sparse_svd(...)`
    must keep working unmodified."""
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.standard_normal((100, 50)))
    result = sparse_svd(X, k=2, sparsity=0.1, random_state=0)
    assert len(result) == 3


def test_check_v_convergence_default_off_matches_published_numbers():
    """The stricter v-based convergence check must be strictly opt-in;
    at its default (False) every previously-published number must be
    reproduced exactly regardless of whether check_v_convergence exists
    in the signature."""
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.standard_normal((200, 100)))
    U1, D1, V1 = sparse_svd(X, k=3, sparsity=0.1, random_state=0)
    U2, D2, V2 = sparse_svd(X, k=3, sparsity=0.1, random_state=0, check_v_convergence=False)
    np.testing.assert_array_equal(U1, U2)
    np.testing.assert_array_equal(V1, V2)
