"""
Synthetic data generators used in the SLICE paper's benchmarks and in
this package's test suite. Ported verbatim from the paper's notebooks
(Test_suite_SLICE.ipynb) so that published numbers remain reproducible.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def make_overlapping_programs(
    n_cells: int = 2000,
    n_genes: int = 2000,
    n_progs: int = 10,
    module_size: int = 50,
    overlap_frac: float = 0.0,
    snr: float = 5.0,
    seed: int = 0,
):
    """Synthetic data with `n_progs` programs of `module_size` genes
    each, sharing `overlap_frac` of their genes with every other
    program. Ground truth gene weights (W) contain both signs.

    Returns
    -------
    X : sparse (n_cells, n_genes) float32 matrix
    S : (n_cells, n_progs) ground-truth cell activities
    W : (n_genes, n_progs) ground-truth signed gene weights
    """
    rng = np.random.default_rng(seed)
    n_shared = int(module_size * overlap_frac)
    n_unique = module_size - n_shared
    W = np.zeros((n_genes, n_progs))
    idx = 0
    for p in range(n_progs):
        W[idx:idx + n_shared, p] = rng.choice([-1, 1], size=n_shared) * rng.uniform(0.5, 1.5, n_shared)
    idx += n_shared
    for p in range(n_progs):
        n_pos = n_unique // 2
        n_neg = n_unique - n_pos
        W[idx:idx + n_pos, p] = rng.uniform(0.5, 1.5, n_pos)
        W[idx + n_pos:idx + n_unique, p] = rng.uniform(-1.5, -0.5, n_neg)
        idx += n_unique
    W = W / np.linalg.norm(W, axis=0)

    S = np.zeros((n_cells, n_progs))
    for p in range(n_progs):
        active = rng.choice(n_cells, int(n_cells * 0.2), replace=False)
        S[active, p] = rng.uniform(1.0, 3.0, len(active))

    signal = S @ W.T
    noise_scale = np.linalg.norm(signal) / np.linalg.norm(np.ones((n_cells, n_genes))) / snr
    X = signal + rng.standard_normal((n_cells, n_genes)) * noise_scale
    return sp.csr_matrix(X.astype(np.float32)), S, W


def make_antagonistic_axis(n_cells: int = 2000, n_genes: int = 1000, snr: float = 5.0, seed: int = 42):
    """A single continuous axis with 50 up-regulated and 50
    down-regulated genes -- the minimal case a non-negative model
    (NMF) cannot represent with one component."""
    rng = np.random.default_rng(seed)
    S_true = np.linspace(-1.5, 1.5, n_cells).reshape(-1, 1)
    W_true = np.zeros((n_genes, 1))
    W_true[:50, 0] = rng.uniform(0.5, 1.5, 50)
    W_true[50:100, 0] = rng.uniform(-1.5, -0.5, 50)
    W_true /= np.linalg.norm(W_true)
    signal = S_true @ W_true.T
    noise_scale = np.linalg.norm(signal) / np.linalg.norm(np.ones((n_cells, n_genes))) / snr
    X = signal + rng.standard_normal((n_cells, n_genes)) * noise_scale
    return sp.csr_matrix(X.astype(np.float32)), S_true, W_true


def make_hard_overlap_counts(seed: int = 42):
    """Poisson-count-scale synthetic data with two hard-overlapping
    cell populations (A: cells 0-6000, B: cells 4000-9000), used for
    the ablation study. Returns raw counts X, the log1p-centered
    matrix Xc, ground-truth cell indicators (tA, tB), and ground-truth
    gene-set indicators (gA, gB)."""
    rng = np.random.default_rng(seed)
    n_cells, n_genes = 10000, 1000
    X = rng.poisson(0.1, (n_cells, n_genes)).astype(np.float32)
    X[:6000, 0:150] += rng.poisson(5, (6000, 150))          # A unique
    X[:6000, 150:250] += rng.poisson(5, (6000, 100))        # shared
    X[4000:9000, 250:400] += rng.poisson(5, (5000, 150))    # B unique
    X[4000:9000, 150:250] += rng.poisson(5, (5000, 100))    # shared

    tA = (np.arange(n_cells) < 6000).astype(float)
    tB = ((np.arange(n_cells) >= 4000) & (np.arange(n_cells) < 9000)).astype(float)
    gA = np.zeros(n_genes); gA[:250] = 1
    gB = np.zeros(n_genes); gB[150:400] = 1

    Xlog = np.log1p(X)
    Xc = sp.csr_matrix(Xlog - Xlog.mean(0))
    return sp.csr_matrix(X), Xc, tA, tB, gA, gB
