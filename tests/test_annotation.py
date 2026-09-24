import warnings

import numpy as np
import pandas as pd
import pytest

from slice_lca.core import SLICEResult
from slice_lca.annotation import load_reference_db, annotate_modules, export_hub_genes


def _make_fake_reference_csv(tmp_path, n_celltypes=120, universe_genes=None, seed=0, with_species=True):
    """A synthetic CellMarker-like CSV: random cell types with random
    marker gene sets drawn from `universe_genes`."""
    rng = np.random.default_rng(seed)
    universe_genes = universe_genes or [f"GENE{i}" for i in range(2000)]
    rows = []
    for ct in range(n_celltypes):
        k = rng.integers(10, 150)
        genes = rng.choice(universe_genes, size=k, replace=False)
        row = {
            "cell_name": f"CellType{ct}",
            "Symbol": ",".join(genes),
        }
        if with_species:
            row["Species"] = "Human"
        rows.append(row)
    df = pd.DataFrame(rows)
    path = tmp_path / "fake_reference.csv"
    df.to_csv(path, index=False)
    return path


def _make_synthetic_result(n_genes=2000, k=3, seed=0):
    rng = np.random.default_rng(seed)
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    kME = rng.uniform(-0.2, 0.2, size=(n_genes, k))
    labels = rng.integers(0, k, size=n_genes)
    U = rng.standard_normal((100, k))
    V = rng.standard_normal((n_genes, k))
    D = np.ones(k)
    return SLICEResult(U=U, D=D, V=V, kME=kME, labels=labels, gene_names=gene_names, k=k)


def test_load_reference_db_case_normalizes_cell_types(tmp_path):
    rng = np.random.default_rng(0)
    universe = [f"GENE{i}" for i in range(200)]
    genes_a = ",".join(rng.choice(universe, 20, replace=False))
    genes_b = ",".join(rng.choice(universe, 20, replace=False))
    df = pd.DataFrame([
        {"cell_name": "T cell", "Symbol": genes_a, "Species": "Human"},
        {"cell_name": "T Cell", "Symbol": genes_b, "Species": "Human"},   # different casing
        {"cell_name": " t cell ", "Symbol": genes_a, "Species": "Human"},  # whitespace + case
    ])
    path = tmp_path / "ref.csv"
    df.to_csv(path, index=False)
    db = load_reference_db(str(path), species="Human", verbose=False)
    # All three rows should merge into ONE cell-type entry, not three.
    matching_keys = [k for k in db if k.strip().lower() == "t cell"]
    assert len(matching_keys) == 1


def test_load_reference_db_warns_when_species_column_missing(tmp_path):
    df = pd.DataFrame([{"cell_name": "T cell", "Symbol": "GENE1,GENE2,GENE3"}])
    path = tmp_path / "ref_no_species.csv"
    df.to_csv(path, index=False)
    with pytest.warns(UserWarning, match="species"):
        load_reference_db(str(path), species="Human", verbose=False)


def test_load_reference_db_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_reference_db("/nonexistent/path.csv", verbose=False)


def test_annotate_modules_recovers_true_enrichment(tmp_path):
    """A module built almost entirely from one cell type's marker genes
    should be identified as that cell type."""
    rng = np.random.default_rng(0)
    universe = [f"GENE{i}" for i in range(2000)]
    target_markers = list(rng.choice(universe, 60, replace=False))

    rows = [{"cell_name": "TargetCell", "Symbol": ",".join(target_markers), "Species": "Human"}]
    for ct in range(50):
        genes = rng.choice(universe, rng.integers(10, 100), replace=False)
        rows.append({"cell_name": f"Decoy{ct}", "Symbol": ",".join(genes), "Species": "Human"})
    df = pd.DataFrame(rows)
    path = tmp_path / "ref.csv"
    df.to_csv(path, index=False)
    db = load_reference_db(str(path), species="Human", verbose=False)

    result = _make_synthetic_result(n_genes=2000, k=2, seed=1)
    # Force module 0's top genes to BE the target cell's markers.
    for g in target_markers:
        gi = result.gene_names.index(g)
        result.labels[gi] = 0
        result.kME[gi, 0] = 0.9

    ann = annotate_modules(result, db, top_n=50, q_thresh=0.05)
    m0 = ann[ann["module"] == "M0"].iloc[0]
    assert m0["predicted_identity"] == "TargetCell"
    assert m0["q_value"] < 0.05


def test_no_false_positive_identification_on_noise(tmp_path):
    """Regression test for the main bug: with a realistically-sized
    reference DB (120 cell types) and a module with NO true
    enrichment for anything, BH-corrected annotation should call it
    'Unknown' far more reliably than a raw p<1e-3 cutoff would.
    Uncorrected, this scenario has an empirical ~5% false-positive
    rate per module (see docstring in annotation.py)."""
    path = _make_fake_reference_csv(tmp_path, n_celltypes=120, seed=1)
    db = load_reference_db(str(path), species="Human", verbose=False)

    n_false_positives = 0
    n_trials = 40
    for seed in range(n_trials):
        result = _make_synthetic_result(n_genes=2000, k=1, seed=100 + seed)
        ann = annotate_modules(result, db, top_n=50, q_thresh=0.05)
        if ann.iloc[0]["predicted_identity"] != "Unknown":
            n_false_positives += 1
    # BH correction should keep this near the nominal FDR level, not
    # anywhere near the ~5%-per-test-times-many-tests rate an
    # uncorrected cutoff would give across repeated random draws.
    assert n_false_positives / n_trials <= 0.15


def test_export_hub_genes_missing_module_warns_not_crashes():
    result = _make_synthetic_result(n_genes=200, k=2, seed=0)
    annotation_df = pd.DataFrame([{"module": "M0", "predicted_identity": "SomeCell"}])  # M1 missing
    with pytest.warns(UserWarning, match="M1"):
        out = export_hub_genes(result, annotation_df, path="/tmp/_test_hub_genes.tsv")
    assert (out.loc[out["module"] == "M1", "predicted_identity"] == "Unknown").all()
