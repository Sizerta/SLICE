"""
slice_lca.bio_outputs
The outputs biologists expect from a WGCNA-style run, built on top of a
SLICEResult (see slice_lca.core.fit_slice):
  - module eigengene <-> trait correlation table (BH-adjusted p-values)
  - hub gene tables (signed kME, the same statistic WGCNA reports)
  - percent variance explained per module
  - a small hub-gene correlation network per module, exportable as a
    Cytoscape edge list (analogous to WGCNA's exportNetworkToCytoscape)
  - module-trait heatmap and variance-explained bar plot

Fixed relative to the original draft (see CHANGELOG.md):
  - `from lca_core import bh_fdr` pointed at a module that doesn't
    exist in this package; bh_fdr now lives in slice_lca.core and is
    imported from there. This was a hard ImportError -- the file
    could not be imported at all before this fix.
  - export_hub_network() densifies only the small (top_n x top_n)
    submatrix it actually needs before calling np.corrcoef. The
    original passed a scipy sparse slice straight to np.corrcoef,
    which doesn't accept sparse input and raised AttributeError.
  - plot_module_trait_heatmap() sorts modules numerically ("M2" before
    "M10") instead of lexicographically ("M10" before "M2"). The
    lexicographic bug is invisible for k<10 and wrong for every k>=10
    module set -- which includes the paper's own NSCLC run (k=20).
  - gene_module_profile() accepts gene_names as a list OR array, does
    case-insensitive gene lookup (consistent with annotation.py, which
    upper-cases everything), and raises a clear error naming the gene
    that wasn't found instead of a bare pandas/numpy exception.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr

from .core import bh_fdr


def variance_explained(D: np.ndarray, X) -> np.ndarray:
    """Fraction of total sum-of-squares in X (||X||_F^2) captured by
    each component: D[m]^2 / ||X||_F^2.

    Caveat: SLICE's components are NOT enforced to be exactly
    orthogonal (deflation only approximately removes prior signal
    under the sparse projection -- see the paper's "Component
    deflation and latent overlap" section). Treat each entry as a
    standalone, WGCNA-style "how much does this module alone explain"
    number; the entries are not guaranteed to sum to the variance
    explained by the joint k-component reconstruction.
    """
    if sp.issparse(X):
        total_var = sp.linalg.norm(X, "fro") ** 2
    else:
        total_var = np.linalg.norm(X, "fro") ** 2
    return (np.asarray(D) ** 2) / total_var


def module_trait_table(U: np.ndarray, trait: np.ndarray, module_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Pearson correlation of each module's cell activity (eigengene)
    with a continuous trait, BH-corrected across modules."""
    k = U.shape[1]
    module_names = list(module_names) if module_names is not None else [f"M{i}" for i in range(k)]
    rows = []
    for m in range(k):
        r, p = pearsonr(U[:, m], trait)
        rows.append((module_names[m], r, p))
    df = pd.DataFrame(rows, columns=["module", "r", "p"])
    df["p_BH"] = bh_fdr(df["p"].values)
    return df.sort_values("p_BH").reset_index(drop=True)


def hub_gene_table(kME: np.ndarray, gene_names: Sequence[str], labels: np.ndarray, module_id: int, top_n: int = 10) -> pd.DataFrame:
    idx = np.where(labels == module_id)[0]
    if len(idx) == 0:
        return pd.DataFrame(columns=["gene", "kME", "sign"])
    vals = kME[idx, module_id]
    order = idx[np.argsort(-np.abs(vals))][:top_n]
    out = pd.DataFrame({
        "gene": [gene_names[i] for i in order],
        "kME": kME[order, module_id],
    })
    out["sign"] = np.where(out["kME"] > 0, "+", "-")
    return out.reset_index(drop=True)


def export_hub_network(
    X, kME: np.ndarray, labels: np.ndarray, gene_names: Sequence[str],
    module_id: int, top_n: int = 30, corr_threshold: float = 0.3,
) -> pd.DataFrame:
    """Cytoscape-ready edge list restricted to a module's top hub
    genes: weight = Pearson correlation between the two genes'
    expression profiles. Cheap because it only touches top_n genes,
    not the full gene set (unlike WGCNA's TOM, which is genome-wide).

    X may be sparse or dense; only the (n_cells x top_n) submatrix is
    ever densified, which is cheap for realistic top_n.
    """
    idx = np.where(labels == module_id)[0]
    if len(idx) == 0:
        return pd.DataFrame(columns=["source", "target", "weight"])
    vals = kME[idx, module_id]
    top = idx[np.argsort(-np.abs(vals))][:top_n]

    if sp.issparse(X):
        sub = X.tocsc()[:, top].toarray()
    else:
        sub = np.asarray(X)[:, top]

    C = np.corrcoef(sub, rowvar=False)
    rows = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            w = C[i, j]
            if abs(w) >= corr_threshold:
                rows.append((gene_names[top[i]], gene_names[top[j]], w))
    return pd.DataFrame(rows, columns=["source", "target", "weight"]).sort_values(
        "weight", key=abs, ascending=False
    ).reset_index(drop=True)


def gene_module_profile(kME: np.ndarray, gene_names: Sequence[str], gene, module_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Full per-module correlation profile for one gene, not just its
    argmax module. Use this when a gene lands somewhere surprising: a
    large *second*-best |kME| means the gene genuinely straddles two
    axes -- a real feature of sequential/deflation-based factorization
    (the strongest, highest-dynamic-range genes tend to anchor
    whichever axis is extracted first, and only their residual shows
    up on later, finer axes) rather than a labelling error.

    Gene lookup is case-insensitive (consistent with annotation.py).
    """
    if isinstance(gene, (int, np.integer)):
        j = int(gene)
    else:
        names_list = list(gene_names)
        gene_upper = str(gene).strip().upper()
        upper_names = [str(g).strip().upper() for g in names_list]
        try:
            j = upper_names.index(gene_upper)
        except ValueError:
            raise ValueError(
                f"Gene {gene!r} not found among {len(names_list)} gene names "
                "(case-insensitive match attempted)."
            ) from None

    k = kME.shape[1]
    module_names = list(module_names) if module_names is not None else [f"M{m}" for m in range(k)]
    vals = kME[j, :]
    order = np.argsort(-np.abs(vals))
    return pd.DataFrame({
        "module": [module_names[m] for m in order],
        "kME": vals[order],
    })


def _module_sort_key(module_label: str):
    """Numeric sort key for labels like 'M0', 'M1', ..., 'M10' so they
    sort as 0, 1, ..., 10 rather than lexicographically ('M10' < 'M2')."""
    digits = "".join(ch for ch in str(module_label) if ch.isdigit())
    return int(digits) if digits else module_label


def plot_module_trait_heatmap(trait_table: pd.DataFrame, trait_name: str = "trait", path: str = "module_trait_heatmap.png") -> str:
    import matplotlib.pyplot as plt

    df = trait_table.copy()
    df["_sort_key"] = df["module"].map(_module_sort_key)
    df = df.sort_values("_sort_key")

    fig, ax = plt.subplots(figsize=(3.2, max(2.0, 0.45 * len(df))))
    r = df["r"].values.reshape(-1, 1)
    im = ax.imshow(r, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["module"])
    ax.set_xticks([0])
    ax.set_xticklabels([trait_name])
    for i, row in enumerate(df.itertuples()):
        stars = "***" if row.p_BH < 0.001 else "**" if row.p_BH < 0.01 else "*" if row.p_BH < 0.05 else ""
        ax.text(0, i, f"{row.r:+.2f}\n{stars}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Pearson r", fraction=0.15)
    ax.set_title("Module-trait relationships")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_variance_explained(ve: np.ndarray, path: str = "variance_explained.png") -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 3))
    modules = [f"M{i}" for i in range(len(ve))]
    ax.bar(modules, np.asarray(ve) * 100, color="#4C72B0")
    ax.set_ylabel("% variance explained")
    ax.set_title("Variance explained per module")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
