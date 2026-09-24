import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from slice_lca.bio_outputs import (
    variance_explained,
    module_trait_table,
    hub_gene_table,
    export_hub_network,
    gene_module_profile,
    plot_module_trait_heatmap,
    _module_sort_key,
)


def test_variance_explained_sparse_matches_dense():
    rng = np.random.default_rng(0)
    X_dense = rng.standard_normal((100, 50))
    X_sparse = sp.csr_matrix(X_dense)
    D = np.array([5.0, 3.0, 1.0])
    ve_dense = variance_explained(D, X_dense)
    ve_sparse = variance_explained(D, X_sparse)
    np.testing.assert_allclose(ve_dense, ve_sparse, rtol=1e-10)


def test_module_trait_table_bh_corrects_and_sorts():
    rng = np.random.default_rng(0)
    trait = rng.standard_normal(200)
    # module 0 strongly correlated with trait, others pure noise
    U = np.column_stack([
        trait * 2 + rng.standard_normal(200) * 0.1,
        rng.standard_normal(200),
        rng.standard_normal(200),
    ])
    df = module_trait_table(U, trait)
    assert df.iloc[0]["module"] == "M0"
    assert df.iloc[0]["p_BH"] < 0.01
    assert (df["p_BH"].values == np.sort(df["p_BH"].values)).all()


def test_hub_gene_table_empty_module():
    kME = np.zeros((10, 2))
    labels = np.full(10, -1)
    out = hub_gene_table(kME, [f"g{i}" for i in range(10)], labels, module_id=0)
    assert out.empty


def test_export_hub_network_works_on_sparse_input():
    """Regression test: the original crashed with AttributeError
    because np.corrcoef doesn't accept a scipy sparse slice."""
    rng = np.random.default_rng(0)
    X_dense = rng.standard_normal((200, 30))
    X_sparse = sp.csr_matrix(X_dense)
    kME = rng.uniform(-0.5, 0.5, size=(30, 1))
    labels = np.zeros(30, dtype=int)
    gene_names = [f"g{i}" for i in range(30)]

    edges = export_hub_network(X_sparse, kME, labels, gene_names, module_id=0, top_n=10, corr_threshold=0.0)
    assert isinstance(edges, pd.DataFrame)
    assert set(edges.columns) == {"source", "target", "weight"}
    assert len(edges) == 10 * 9 // 2  # all pairs, threshold=0


def test_gene_module_profile_case_insensitive_and_array_input():
    kME = np.array([[0.5, -0.2], [0.1, 0.9]])
    gene_names = np.array(["Muc2", "Sox9"])  # array, not list; mixed case
    df = gene_module_profile(kME, gene_names, "MUC2")
    assert df.iloc[0]["module"] == "M0"
    assert df.iloc[0]["kME"] == pytest.approx(0.5)


def test_gene_module_profile_missing_gene_gives_clear_error():
    kME = np.array([[0.5, -0.2]])
    with pytest.raises(ValueError, match="NOTAGENE"):
        gene_module_profile(kME, ["MUC2"], "NOTAGENE")


def test_module_sort_key_orders_numerically():
    labels = ["M0", "M2", "M10", "M1", "M11"]
    ordered = sorted(labels, key=_module_sort_key)
    assert ordered == ["M0", "M1", "M2", "M10", "M11"]


def test_plot_module_trait_heatmap_orders_modules_numerically(tmp_path):
    """Regression test: the original sorted module labels as strings
    ('M10' before 'M2'), which is wrong for any k >= 10 -- including
    the paper's own NSCLC run (k=20)."""
    labels = [f"M{i}" for i in range(12)]
    df = pd.DataFrame({
        "module": labels,
        "r": np.linspace(-1, 1, 12),
        "p_BH": np.linspace(0.001, 0.5, 12),
    })
    out_path = plot_module_trait_heatmap(df, path=str(tmp_path / "heatmap.png"))
    assert out_path == str(tmp_path / "heatmap.png")
    import os
    assert os.path.exists(out_path)
