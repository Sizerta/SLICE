"""
slice_lca.annotation
WGCNA-style biological validation and automatic annotation for SLICE
modules, using a marker-gene reference database (e.g. CellMarker,
PanglaoDB) and a hypergeometric enrichment test.

Fixed relative to the original draft (see CHANGELOG.md):
  - annotate_modules() now applies Benjamini-Hochberg FDR correction
    across ALL (module x cell-type) tests, not a raw per-test p-value
    cutoff. Uncorrected, testing each module against a ~100+ cell-type
    reference database at p<1e-3 gives a ~5% chance of confidently
    "identifying" a module that has no real enrichment at all (see
    tests/test_annotation.py::test_no_false_positive_identification_on_noise,
    which reproduces this empirically). This was the single most
    important fix: it directly determines whether the module names in
    a Results table are trustworthy.
  - load_reference_db() now prints which column it selected for each
    role (cell type, gene symbol, marker alias, species), and warns
    instead of silently skipping if species filtering was requested
    but no species column was found. Silent column auto-detection is
    the main way this kind of tool goes wrong without anyone noticing.
  - cell-type labels are case/whitespace-normalized before grouping,
    so "T cell", "T Cell", " T cell" no longer become three separate,
    smaller reference sets.
  - plot_module_marker_heatmap() no longer calls list.index() inside a
    double loop (O(n_genes) per lookup); it precomputes a gene->index
    dict once. On a 20,000-gene panel this was the dominant cost.
  - export_hub_genes() no longer crashes with an opaque IndexError if
    a module is missing from annotation_df; it fills in "Unknown" and
    prints a warning instead.
"""
from __future__ import annotations

import os
import warnings
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from .core import bh_fdr


def load_reference_db(
    filepath: str,
    species: str = "Human",
    cancer_type: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Set[str]]:
    """Load a marker-gene reference database (CellMarker-style export)
    into {cell_type: {GENE, ...}}.

    Column names are auto-detected by keyword. This is inherently a
    little fragile across different database exports, so with
    verbose=True (default) the function prints exactly which column it
    picked for each role -- check this output the first time you point
    it at a new file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Database file not found at {filepath}")

    if filepath.endswith((".xlsx", ".xls")):
        if verbose:
            print("Reading Excel file (this may take ~20-30 seconds for CellMarker)...")
        df = pd.read_excel(filepath, engine="openpyxl")
    else:
        delim = "," if filepath.endswith(".csv") else "\t"
        df = pd.read_csv(filepath, sep=delim, low_memory=False)

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    def find_col(keywords):
        for kw in keywords:
            for c in df.columns:
                if kw in c:
                    return c
        return None

    col_species = find_col(["species"])
    col_cell = find_col(["cell_name", "celltype", "cell_type"])
    col_symbol = find_col(["symbol", "official_gene_symbol", "gene"])
    col_marker = find_col(["marker"])

    if verbose:
        print(
            "Column selection -- cell type: "
            f"{col_cell!r}, gene symbol: {col_symbol!r}, marker alias: "
            f"{col_marker!r}, species: {col_species!r}. If any of these "
            "look wrong, check for a naming collision in your file's "
            "columns and pass a pre-filtered file instead."
        )

    if col_cell is None or (col_symbol is None and col_marker is None):
        raise ValueError(f"Could not find cell/gene columns. Found: {df.columns.tolist()}")

    if species.lower() != "all":
        if col_species is None:
            warnings.warn(
                f"species={species!r} was requested but no species column "
                "was found -- NO species filtering was applied. The "
                "reference database may contain a mix of species.",
                stacklevel=2,
            )
        else:
            sp_lower = species.lower()
            targets = (
                ["human", "homo sapiens"] if "human" in sp_lower
                else ["mouse", "mus musculus"] if "mouse" in sp_lower
                else [sp_lower]
            )
            col_vals = df[col_species].astype(str).str.lower()
            mask = np.zeros(len(df), dtype=bool)
            for t in targets:
                mask |= col_vals.str.contains(t, na=False)
            n_before = len(df)
            df = df[mask]
            if verbose:
                print(f"Species filter ({species}): kept {len(df)}/{n_before} rows.")

    if cancer_type is not None:
        col_cancer = find_col(["cancer_type"])
        if col_cancer:
            df = df[df[col_cancer].astype(str).str.contains(cancer_type, case=False, na=False)]
        elif verbose:
            warnings.warn(
                f"cancer_type={cancer_type!r} was requested but no cancer_type "
                "column was found -- filter not applied.",
                stacklevel=2,
            )

    # Normalize cell-type labels so "T cell" / "T Cell" / " T cell " merge.
    cells = df[col_cell].astype(str).str.strip()
    df = df.assign(_cell=cells)
    df = df[df["_cell"].str.len() > 0]
    # Keep a display-cased version but group on a normalized key.
    df["_cell_key"] = df["_cell"].str.lower()

    def exploded(use_col):
        s = df[use_col].astype(str).str.strip().str.upper()
        s = s.str.replace("|", ",", regex=False).str.split(",")
        out = df[["_cell", "_cell_key"]].copy()
        out["_g"] = s
        return out.explode("_g")

    frames = []
    if col_symbol:
        frames.append(exploded(col_symbol))
    if col_marker:
        frames.append(exploded(col_marker))
    combined = pd.concat(frames, ignore_index=True)

    combined["_g"] = combined["_g"].astype(str).str.strip().str.upper()
    combined = combined[combined["_g"].str.len() > 0]
    combined = combined[combined["_g"] != "NAN"]

    db_dict: Dict[str, Set[str]] = {}
    for key, g in combined.groupby("_cell_key"):
        # Use the most common original-cased spelling as the display name.
        display_name = g["_cell"].mode().iloc[0]
        db_dict[display_name] = set(g["_g"].unique())
    db_dict = {ct: genes for ct, genes in db_dict.items() if len(genes) >= 3}

    if verbose:
        print(f"Successfully loaded {len(db_dict)} cell types ({sum(len(v) for v in db_dict.values()):,} gene entries).")
        top = sorted(db_dict.items(), key=lambda kv: -len(kv[1]))[:5]
        print("  largest sets:", ", ".join(f"{ct} ({len(g)})" for ct, g in top))
    return db_dict


def annotate_modules(
    result,
    db_dict: Dict[str, Set[str]],
    top_n: int = 50,
    min_overlap: int = 2,
    min_set: int = 3,
    max_set_frac: float = 0.25,
    q_thresh: float = 0.05,
) -> pd.DataFrame:
    """Annotate each module with its best-matching cell type from
    `db_dict`, via hypergeometric enrichment of the module's top_n
    hub genes (ranked by |kME|) against each reference gene set.

    Multiple-testing correction: with a reference database of ~100+
    cell types, taking the single best p-value per module and
    comparing it to a fixed uncorrected threshold gives a
    surprisingly high false-positive rate purely from testing many
    cell types at once (empirically ~5% per module at p<1e-3 against
    a 120-cell-type reference with ZERO true signal -- see
    tests/test_annotation.py). This function instead applies
    Benjamini-Hochberg correction across every (module, cell type)
    test performed in this call, and reports a module as identified
    only if its best BH-adjusted q-value is below `q_thresh`.
    """
    universe = set(g.upper() for g in result.gene_names)
    M = len(universe)

    # Pass 1: collect every (module, cell_type) test that clears the
    # basic filters, so we can BH-correct across the whole family.
    candidates = []  # (module_idx, cell_type, x, n, K, fold, p)
    module_top_genes = {}
    for m in range(result.k):
        idx = np.where(result.labels == m)[0]
        if len(idx) == 0:
            module_top_genes[m] = set()
            continue
        vals = result.kME[idx, m]
        order = idx[np.argsort(-np.abs(vals))][:top_n]
        top_genes = set(result.gene_names[i].upper() for i in order)
        module_top_genes[m] = top_genes
        n = len(top_genes)
        for ct, ref in db_dict.items():
            ref_in = ref & universe
            K = len(ref_in)
            if K < min_set or K > max_set_frac * M:
                continue
            x = len(top_genes & ref_in)
            if x < min_overlap:
                continue
            fold = (x / n) / (K / M) if n > 0 else 0.0
            p = hypergeom.sf(x - 1, M, K, n)
            candidates.append((m, ct, x, n, K, fold, p))

    rows = []
    if candidates:
        pvals = np.array([c[6] for c in candidates])
        qvals = bh_fdr(pvals)
        best_per_module: Dict[int, tuple] = {}
        for (m, ct, x, n, K, fold, p), q in zip(candidates, qvals):
            cur = best_per_module.get(m)
            # Rank by q first, break ties by higher fold-enrichment.
            if cur is None or (q, -fold) < (cur[0], -cur[2]):
                best_per_module[m] = (q, ct, fold, x, p)
    else:
        best_per_module = {}

    for m in range(result.k):
        n_genes_in_module = int((result.labels == m).sum())
        idx = np.where(result.labels == m)[0]
        if len(idx) > 0:
            vals = result.kME[idx, m]
            top5_idx = idx[np.argsort(-np.abs(vals))][:5]
            top5_display = ", ".join(result.gene_names[i] for i in top5_idx)
        else:
            top5_display = ""
        best = best_per_module.get(m)
        if best is not None and best[0] < q_thresh:
            q, ct, fold, x, p = best
            pred, n_ov = ct, x
        else:
            q, fold, p = np.nan, 0.0, np.nan
            pred, n_ov = "Unknown", 0
        rows.append(dict(
            module=f"M{m}",
            n_genes=n_genes_in_module,
            top_genes=top5_display,
            predicted_identity=pred,
            n_overlap=n_ov,
            fold_enrichment=round(fold, 1),
            p_value=p,
            q_value=q,
        ))
    return pd.DataFrame(rows)


def plot_module_marker_heatmap(result, marker_dict: Dict[str, list], path: str = "module_marker_heatmap.png"):
    """Heatmap of kME for a curated set of marker genes across all
    modules. Uses a precomputed gene->index dict (O(1) lookup) rather
    than list.index() inside a nested loop -- on a 20,000-gene panel
    the original was easily the slowest part of the annotation step."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    gene_list = [g.upper() for g in result.gene_names]
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}  # last occurrence wins if dupes

    plot_kmes, y_labels = [], []
    for ct, genes in marker_dict.items():
        for g in genes:
            g_up = g.upper()
            if g_up in gene_to_idx:
                idx = gene_to_idx[g_up]
                plot_kmes.append(result.kME[idx, :])
                y_labels.append(f"{g_up} ({ct})")

    if not plot_kmes:
        print("No marker genes found in the dataset.")
        return None

    kme_matrix = np.array(plot_kmes)
    fig, ax = plt.subplots(figsize=(max(10, result.k * 1.0), max(4, len(y_labels) * 0.4)))
    sns.heatmap(
        kme_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
        xticklabels=[f"M{i}" for i in range(result.k)],
        yticklabels=y_labels, ax=ax, vmin=-1, vmax=1,
        cbar_kws={"label": "kME (Pearson r)"},
    )
    ax.set_title("Module Eigengene Correlations (kME) for Marker Genes")
    ax.set_xlabel("Module")
    ax.set_ylabel("Marker Gene (Cell Type)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved marker heatmap to {path}")
    return kme_matrix


def export_hub_genes(result, annotation_df: pd.DataFrame, top_n: int = 20, path: str = "hub_genes_export.tsv") -> pd.DataFrame:
    """Per-module hub gene table with the module's predicted identity
    attached. Falls back to 'Unknown' (with a warning) instead of
    crashing if a module is missing from annotation_df."""
    id_lookup = dict(zip(annotation_df["module"], annotation_df["predicted_identity"]))
    rows = []
    for m in range(result.k):
        idx = np.where(result.labels == m)[0]
        if len(idx) == 0:
            continue
        vals = result.kME[idx, m]
        order = idx[np.argsort(-np.abs(vals))][:top_n]
        module_key = f"M{m}"
        pred = id_lookup.get(module_key)
        if pred is None:
            warnings.warn(f"{module_key} not found in annotation_df; labeling as 'Unknown'.", stacklevel=2)
            pred = "Unknown"
        for rank, i in enumerate(order):
            rows.append({
                "module": module_key, "predicted_identity": pred, "rank": rank + 1,
                "gene": result.gene_names[i], "kME": round(float(result.kME[i, m]), 4),
                "sign": "+" if result.kME[i, m] > 0 else "-",
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)
    print(f"Exported {len(df)} hub genes to {path}")
    return df
