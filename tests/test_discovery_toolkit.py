import numpy as np
import scipy.sparse as sp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from discovery_toolkit import (
    cross_tissue_module_reproducibility,
    held_out_marker_recovery,
)
from slice_lca.core import sparse_svd, canonicalize_signs, compute_kME, assign_modules, SLICEResult


def test_cross_tissue_reproducibility_finds_shared_program():
    n_genes = 500
    shared_genes = list(range(0, 40))

    def make_dataset(private_offset, seed):
        rng = np.random.default_rng(seed)
        n_cells = 800
        X = rng.poisson(0.1, (n_cells, n_genes)).astype(np.float32)
        active = rng.choice(n_cells, n_cells // 2, replace=False)
        X[np.ix_(active, shared_genes)] += rng.poisson(5, (len(active), len(shared_genes)))
        priv = list(range(private_offset, private_offset + 40))
        active2 = rng.choice(n_cells, n_cells // 3, replace=False)
        X[np.ix_(active2, priv)] += rng.poisson(5, (len(active2), len(priv)))
        Xlog = np.log1p(X)
        Xc = sp.csr_matrix(Xlog - Xlog.mean(0))
        gene_names = [f"GENE{i}" for i in range(n_genes)]
        return Xc, gene_names

    results = {}
    for name, offset, seed in [("tissue_A", 100, 1), ("tissue_B", 300, 2)]:
        Xc, gene_names = make_dataset(offset, seed)
        U, D, V = sparse_svd(Xc, k=3, sparsity=0.1, random_state=0)
        U, V = canonicalize_signs(U, V)
        kME = compute_kME(Xc, U)
        labels, _ = assign_modules(kME, min_kME=0.1)
        results[name] = SLICEResult(U=U, D=D, V=V, kME=kME, labels=labels, gene_names=gene_names, k=3)

    df = cross_tissue_module_reproducibility(results, top_n=30)
    assert len(df) > 0
    assert df.iloc[0]["jaccard"] > 0.3


def test_held_out_marker_recovery_finds_novel_co_members():
    rng = np.random.default_rng(0)
    n_cells, n_genes = 1000, 400
    X = rng.poisson(0.1, (n_cells, n_genes)).astype(np.float32)
    active = rng.choice(n_cells, n_cells // 2, replace=False)
    X[np.ix_(active, range(0, 30))] += rng.poisson(5, (len(active), 30))
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    known_markers = {"CellTypeX": [f"GENE{i}" for i in [0, 3, 7, 12, 20]]}

    Xlog = np.log1p(X)
    Xc = sp.csr_matrix(Xlog - Xlog.mean(0))
    U, D, V = sparse_svd(Xc, k=3, sparsity=0.1, random_state=0)
    U, V = canonicalize_signs(U, V)
    kME = compute_kME(Xc, U)
    labels, _ = assign_modules(kME, min_kME=0.1)
    result = SLICEResult(U=U, D=D, V=V, kME=kME, labels=labels, gene_names=gene_names, k=3)

    df = held_out_marker_recovery(result, known_markers)
    assert df.iloc[0]["recall"] > 0.6
    assert df.iloc[0]["n_novel_co_members"] > 0
