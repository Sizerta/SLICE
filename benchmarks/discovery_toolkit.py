"""
Discovery toolkit: three ways to show SLICE finds something beyond
marker recapitulation, ranked by how much I could verify without your
real data.

  (1) cross_tissue_module_reproducibility() -- fully tested here on
      synthetic data with a known shared program. Uses ONLY the real
      datasets you already have; no new data or packages needed.
  (2) held_out_marker_recovery() -- fully tested here on synthetic
      data. Also uses only what you already have.
  (3) go_enrichment_beyond_markers() -- correct against gseapy's real
      API (verified the call signature against the installed
      package), but I cannot test an actual network call to Enrichr
      from this sandbox (no internet route to maayanlab.cloud here).
      Test this one first on a single module in Colab before trusting
      it on everything.

Install in Colab: !pip install gseapy
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# =================================================================
# (1) Cross-tissue module reproducibility
# =================================================================
def cross_tissue_module_reproducibility(
    results: Dict[str, "SLICEResult"],  # noqa: F821 -- see slice_lca.core
    top_n: int = 50,
    min_shared_genes: int = 3,
) -> pd.DataFrame:
    """For every pair of datasets you've already run SLICE on (e.g.
    pancreas, intestine, glioblastoma, breast cancer, NSCLC), find the
    best-matching module pair by top-hub-gene overlap (Jaccard), on
    genes present in BOTH gene panels.

    A module that recurs across unrelated tissues with a consistent
    hub-gene signature is a much stronger discovery claim than
    recovering NEUROG3/SOX9 in one dataset -- it says the axis SLICE
    found isn't a one-off fit to noise in a single dataset.

    Parameters
    ----------
    results : dict of {dataset_name: SLICEResult}
        e.g. {"pancreas": pancreas_result, "intestine": intestine_result, ...}

    Returns
    -------
    DataFrame, one row per (dataset_A, module_A, dataset_B, module_B)
    pair, sorted by Jaccard overlap of their top hub genes,
    restricted to pairs clearing `min_shared_genes`.
    """
    names = list(results.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            res_a, res_b = results[name_a], results[name_b]
            genes_a = [g.upper() for g in res_a.gene_names]
            genes_b_set = set(g.upper() for g in res_b.gene_names)

            hub_sets_a = _top_hub_gene_sets(res_a, top_n)
            hub_sets_b = _top_hub_gene_sets(res_b, top_n)

            for m_a, hub_a in hub_sets_a.items():
                hub_a_in_common = hub_a & genes_b_set
                for m_b, hub_b in hub_sets_b.items():
                    shared = hub_a_in_common & hub_b
                    if len(shared) < min_shared_genes:
                        continue
                    union = hub_a_in_common | hub_b
                    jaccard = len(shared) / len(union) if union else 0.0
                    rows.append({
                        "dataset_A": name_a, "module_A": f"M{m_a}",
                        "dataset_B": name_b, "module_B": f"M{m_b}",
                        "n_shared_genes": len(shared), "jaccard": jaccard,
                        "shared_genes": ", ".join(sorted(shared)[:15]),
                    })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("jaccard", ascending=False).reset_index(drop=True)
    return df


def _top_hub_gene_sets(result, top_n: int) -> Dict[int, set]:
    out = {}
    for m in range(result.k):
        idx = np.where(result.labels == m)[0]
        if len(idx) == 0:
            out[m] = set()
            continue
        vals = result.kME[idx, m]
        order = idx[np.argsort(-np.abs(vals))][:top_n]
        out[m] = set(result.gene_names[i].upper() for i in order)
    return out


def _self_test_cross_tissue():
    """Two synthetic 'tissues' sharing one real program plus their
    own private noise-driven modules; the shared one should come out
    on top by a wide margin."""
    from slice_lca.core import sparse_svd, canonicalize_signs, compute_kME, assign_modules, SLICEResult
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    n_genes = 500
    shared_genes = list(range(0, 40))  # a program present in BOTH tissues

    def make_dataset(private_offset, seed):
        r = np.random.default_rng(seed)
        n_cells = 800
        X = r.poisson(0.1, (n_cells, n_genes)).astype(np.float32)
        active = r.choice(n_cells, n_cells // 2, replace=False)
        X[np.ix_(active, shared_genes)] += r.poisson(5, (len(active), len(shared_genes)))
        priv = list(range(private_offset, private_offset + 40))
        active2 = r.choice(n_cells, n_cells // 3, replace=False)
        X[np.ix_(active2, priv)] += r.poisson(5, (len(active2), len(priv)))
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
    print(df[["dataset_A", "module_A", "dataset_B", "module_B", "n_shared_genes", "jaccard"]].to_string(index=False))
    assert df.iloc[0]["jaccard"] > 0.3, "expected the shared program to dominate the top match"
    return df


# =================================================================
# (2) Held-out marker recovery ("blind" discovery test)
# =================================================================
def held_out_marker_recovery(
    result_with_markers_hidden,  # a SLICEResult fit WITHOUT special treatment of markers
    known_markers: Dict[str, List[str]],
    top_n: int = 50,
) -> pd.DataFrame:
    """The cleanest way to show SLICE finds structure rather than just
    reflecting curated input: fit SLICE normally (markers are NOT
    given any special weight -- they're just genes in the panel), then
    check whether known marker genes end up as each other's top
    co-members purely from unsupervised structure.

    This does NOT require re-fitting with markers removed (there's no
    "supervision" to remove -- SLICE never sees cell-type labels). It
    just asks: of a curated marker list you did NOT use to interpret
    the modules, how many correctly land together?

    Returns a per-cell-type row with recall (fraction of that
    cell type's markers assigned to the SAME single best-matching
    module) -- the real "discovery" evidence is markers you didn't
    curate that also land in that module (see `novel_co_members`).
    """
    gene_upper = [g.upper() for g in result_with_markers_hidden.gene_names]
    gene_to_idx = {g: i for i, g in enumerate(gene_upper)}
    labels = result_with_markers_hidden.labels

    rows = []
    for cell_type, markers in known_markers.items():
        marker_idx = [gene_to_idx[g.upper()] for g in markers if g.upper() in gene_to_idx]
        if len(marker_idx) < 2:
            continue
        marker_modules = labels[marker_idx]
        marker_modules_valid = marker_modules[marker_modules >= 0]
        if len(marker_modules_valid) == 0:
            rows.append({"cell_type": cell_type, "n_markers_in_panel": len(marker_idx),
                         "best_module": None, "recall": 0.0, "n_novel_co_members": 0,
                         "novel_co_members": ""})
            continue
        best_module = np.bincount(marker_modules_valid).argmax()
        recall = float((marker_modules == best_module).sum()) / len(marker_idx)

        all_module_genes = set(i for i in np.where(labels == best_module)[0])
        novel = sorted(set(gene_upper[i] for i in all_module_genes) - set(m.upper() for m in markers))
        rows.append({
            "cell_type": cell_type, "n_markers_in_panel": len(marker_idx),
            "best_module": f"M{best_module}", "recall": round(recall, 3),
            "n_novel_co_members": len(novel), "novel_co_members": ", ".join(novel[:20]),
        })
    return pd.DataFrame(rows)


def _self_test_held_out_markers():
    from slice_lca.core import sparse_svd, canonicalize_signs, compute_kME, assign_modules, SLICEResult
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    n_cells, n_genes = 1000, 400
    X = rng.poisson(0.1, (n_cells, n_genes)).astype(np.float32)
    active = rng.choice(n_cells, n_cells // 2, replace=False)
    X[np.ix_(active, range(0, 30))] += rng.poisson(5, (len(active), 30))
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    known_markers = {"CellTypeX": [f"GENE{i}" for i in [0, 3, 7, 12, 20]]}  # subset of the true 0-30 block

    Xlog = np.log1p(X)
    Xc = sp.csr_matrix(Xlog - Xlog.mean(0))
    U, D, V = sparse_svd(Xc, k=3, sparsity=0.1, random_state=0)
    U, V = canonicalize_signs(U, V)
    kME = compute_kME(Xc, U)
    labels, _ = assign_modules(kME, min_kME=0.1)
    result = SLICEResult(U=U, D=D, V=V, kME=kME, labels=labels, gene_names=gene_names, k=3)

    df = held_out_marker_recovery(result, known_markers)
    print(df.to_string(index=False))
    assert df.iloc[0]["recall"] > 0.6
    assert df.iloc[0]["n_novel_co_members"] > 0  # genes 1,2,4,5,... should also show up
    return df


# =================================================================
# (3) GO / pathway enrichment on module hub genes (NOT network-tested)
# =================================================================
def go_enrichment_beyond_markers(
    result,
    module_id: int,
    known_marker_genes: Optional[Sequence[str]] = None,
    top_n: int = 100,
    gene_sets: Sequence[str] = ("GO_Biological_Process_2021", "KEGG_2021_Human"),
    organism: str = "human",
) -> pd.DataFrame:
    """Runs Enrichr (via gseapy) on a module's top hub genes, EXCLUDING
    any gene you've already used as a canonical marker for
    interpretation -- so a significant hit here is evidence the module
    carries biology beyond the markers you fed it.

    Requires internet access to maayanlab.cloud (fine in Colab; NOT
    reachable from the sandbox this was written in, so test this one
    call yourself before trusting the rest of the pipeline).
    """
    try:
        import gseapy as gp
    except ImportError as e:
        raise ImportError("pip install gseapy") from e

    idx = np.where(result.labels == module_id)[0]
    if len(idx) == 0:
        return pd.DataFrame()
    vals = result.kME[idx, module_id]
    order = idx[np.argsort(-np.abs(vals))][:top_n]
    genes = [result.gene_names[i] for i in order]

    known_upper = set(g.upper() for g in (known_marker_genes or []))
    genes_novel = [g for g in genes if g.upper() not in known_upper]

    if len(genes_novel) < 5:
        warnings.warn(
            f"Module M{module_id}: only {len(genes_novel)} genes remain after "
            "excluding known markers -- too few for a meaningful enrichment test.",
            stacklevel=2,
        )
        return pd.DataFrame()

    enr = gp.enrichr(gene_list=genes_novel, gene_sets=list(gene_sets), organism=organism, outdir=None, no_plot=True)
    out = enr.results.sort_values("Adjusted P-value").reset_index(drop=True)
    out.insert(0, "module", f"M{module_id}")
    out.insert(1, "n_genes_tested", len(genes_novel))
    return out


if __name__ == "__main__":
    print("=== (1) cross-tissue module reproducibility, self-test ===")
    _self_test_cross_tissue()
    print("\n=== (2) held-out marker recovery, self-test ===")
    _self_test_held_out_markers()
    print("\n=== (3) GO enrichment: not self-tested here (needs internet to Enrichr) ===")
    print("Run e.g.:")
    print("  go_enrichment_beyond_markers(pancreas_result, module_id=7,")
    print("      known_marker_genes=['NEUROG3','FEV','SOX9','HNF1B'])")
