"""
Integration tests that pin down the paper's published numbers as
regression tests. If these start failing, either the algorithm
changed behavior (investigate before merging) or the paper's numbers
need to be regenerated and updated together with these assertions.
"""
import numpy as np
import pytest

from slice_lca.core import sparse_svd, canonicalize_signs
from slice_lca.datasets import make_antagonistic_axis, make_hard_overlap_counts


def test_antagonistic_axis_recovered_with_single_component():
    """Paper claim (Results, 'Efficient representation of antagonistic
    biological axes'): SLICE k=1 gives state/gene correlation ~1.0 and
    MSE ~3e-5 on a 50-up/50-down synthetic axis."""
    X, S_true, W_true = make_antagonistic_axis(n_cells=2000, n_genes=1000, snr=5.0, seed=0)
    U, D, V = sparse_svd(X, k=1, sparsity=0.15, n_iter=100, tol=1e-4, random_state=0)
    if np.corrcoef(W_true[:, 0], V[:, 0])[0, 1] < 0:
        V[:, 0], U[:, 0] = -V[:, 0], -U[:, 0]
    state_corr = np.corrcoef(S_true[:, 0], U[:, 0])[0, 1]
    gene_corr = np.corrcoef(W_true[:, 0], V[:, 0])[0, 1]
    assert state_corr > 0.99
    assert gene_corr > 0.99


def test_ablation_full_model_matches_published_table():
    """Regression test for Table 2, 'SLICE (full)' row."""
    X, Xc, tA, tB, gA, gB = make_hard_overlap_counts(seed=42)
    U, D, V = sparse_svd(Xc, k=4, sparsity=0.02, random_state=0)
    U, V = canonicalize_signs(U, V)

    def cs(Umat, t):
        return max(abs(np.corrcoef(t, Umat[:, j])[0, 1]) for j in range(Umat.shape[1]) if np.std(Umat[:, j]) > 0)

    def gs(Vmat, g):
        return max(abs(np.corrcoef(g, Vmat[:, j])[0, 1]) for j in range(Vmat.shape[1]) if np.std(Vmat[:, j]) > 0)

    assert cs(U, tA) == pytest.approx(0.900, abs=1e-3)
    assert cs(U, tB) == pytest.approx(0.999, abs=1e-3)
    assert gs(V, gA) == pytest.approx(0.177, abs=1e-3)
    assert gs(V, gB) == pytest.approx(0.247, abs=1e-3)


def test_deflation_off_produces_redundant_components():
    """Regression test for the corrected ablation finding: without
    deflation (sparsity held fixed), at least two of the four
    components should be near-duplicates (this is the *actual*
    mechanism by which deflation helps in this test -- not a
    cell-state-separation collapse; see PR discussion / paper v3)."""
    X, Xc, tA, tB, gA, gB = make_hard_overlap_counts(seed=42)
    _, _, V_off = sparse_svd(Xc, k=4, sparsity=0.02, deflate=False, random_state=0)
    comp_corr = np.abs(np.corrcoef(V_off.T))
    np.fill_diagonal(comp_corr, 0)
    assert comp_corr.max() > 0.8
