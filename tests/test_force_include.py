import numpy as np
import scipy.sparse as sp

from slice_lca.preprocessing import select_hvg, preprocess


def test_force_include_keeps_low_variance_marker():
    """Reproduces exactly what happened with DCLK1/TRPM5/PTPRC/CD3E:
    a real gene present in the full panel, with low enough variance
    that it doesn't make the top-N cut on its own."""
    rng = np.random.default_rng(0)
    n_cells, n_genes = 500, 200
    X = sp.csr_matrix(rng.poisson(0.3, (n_cells, n_genes)).astype(np.float32))
    gene_names = np.array([f"gene{i}" for i in range(n_genes)])
    gene_names[50] = "DCLK1"  # deliberately low-variance (background level, no boost)

    # boost variance for everything except DCLK1 so it's guaranteed to
    # rank outside the top 20 on variance alone
    X = X.tolil()
    for g in range(n_genes):
        if g != 50:
            X[:, g] = rng.poisson(3, n_cells)
    X = X.tocsr()

    X_no_force, genes_no_force = select_hvg(X, gene_names, n_top=20)
    assert "DCLK1" not in genes_no_force, "test setup invalid -- DCLK1 should be HVG-dropped without force_include"

    X_forced, genes_forced = select_hvg(X, gene_names, n_top=20, force_include=["DCLK1"])
    assert "DCLK1" in genes_forced
    assert X_forced.shape[1] == len(genes_forced)
    assert X_forced.shape[0] == X.shape[0]


def test_force_include_case_insensitive():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1, (100, 50)).astype(np.float32))
    # Build the final names directly -- assigning a longer string into
    # an already-narrower-dtype array element silently truncates it
    # (numpy fixed-width unicode arrays), which is exactly what the
    # first version of this test did by accident.
    names = [f"g{i}" for i in range(50)]
    names[10] = "Sox9"
    gene_names = np.array(names)
    assert gene_names.dtype.kind == "U" and gene_names[10] == "Sox9"  # guard the test itself
    _, genes = select_hvg(X, gene_names, n_top=5, force_include=["SOX9"])
    assert "Sox9" in genes


def test_force_include_ignores_absent_genes_silently():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1, (100, 50)).astype(np.float32))
    gene_names = np.array([f"g{i}" for i in range(50)])
    # "NOTPRESENT" isn't in gene_names at all -- must not raise
    X_out, genes_out = select_hvg(X, gene_names, n_top=5, force_include=["NOTPRESENT"])
    assert "NOTPRESENT" not in genes_out
    assert X_out.shape[1] == len(genes_out)


def test_force_include_none_behaves_exactly_as_before():
    """No regression: force_include=None (the default) must give
    byte-identical results to the pre-existing behavior."""
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1, (200, 80)).astype(np.float32))
    gene_names = np.array([f"g{i}" for i in range(80)])
    X1, g1 = select_hvg(X, gene_names, n_top=10)
    X2, g2 = select_hvg(X, gene_names, n_top=10, force_include=None)
    np.testing.assert_array_equal(g1, g2)
    np.testing.assert_array_equal(X1.toarray(), X2.toarray())


def test_preprocess_force_include_end_to_end():
    rng = np.random.default_rng(0)
    n_cells, n_genes = 400, 150
    X = sp.csr_matrix(rng.poisson(3, (n_cells, n_genes)).astype(np.float32))
    gene_names = [f"g{i}" for i in range(n_genes)]
    gene_names[77] = "PTPRC"
    X = X.tolil()
    X[:, 77] = rng.poisson(0.2, n_cells)  # force PTPRC to be low-variance
    X = X.tocsr()

    X_ready, genes_ready = preprocess(X, gene_names, n_top_hvg=10, skip_qc=True,
                                        force_include=["PTPRC"])
    assert "PTPRC" in [g.upper() if isinstance(g, str) else g for g in genes_ready] or "PTPRC" in list(genes_ready)
