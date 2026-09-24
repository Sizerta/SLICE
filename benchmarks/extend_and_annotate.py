"""
Three things, in the order you'll actually want to run them:

  (1) Extend to all 5 datasets (pancreas + NSCLC added to what you
      already have) -- cross-tissue reproducibility AND cNMF agreement.
  (2) GO enrichment specifically on the myeloid and stromal
      cross-tissue modules -- your strongest finding so far.
  (3) Flag (not silently drop) the immediate-early-gene / dissociation
      match so it doesn't accidentally end up in the manuscript.

Everything here is built on functions already in your
compare_slice_cnmf.py / discovery_toolkit.py -- this just wires them
together for the specific next step. Tested pieces are noted; the
NSCLC-scale runtime note is a real constraint, not a guess -- read it
before kicking off a run you'll be waiting hours for.
"""
import sys
sys.path.insert(0, "slice-lca/benchmarks")
sys.path.insert(0, "slice-lca/src")

import numpy as np
import pandas as pd
import scipy.sparse as sp

from slice_lca.core import fit_slice
from slice_lca.preprocessing import preprocess
import compare_slice_cnmf as cs_cnmf
import discovery_toolkit as dt


# =================================================================
# (1) Extend to pancreas + NSCLC
# =================================================================

# --- Pancreas: same as your existing cell 11, just wrapped ---
def load_pancreas():
    import scvelo as scv
    import gc
    adata = scv.datasets.pancreas()
    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    gene_names = adata.var_names.to_numpy()
    del adata
    gc.collect()
    return X, gene_names


# --- NSCLC: EDIT THIS to however you already load the 897k-cell
# dataset elsewhere in your notebook (the Real_DealV4-style multi-part
# streaming download) -- I don't have that URL, so this is a
# placeholder matching the shape everything downstream expects. ---
def load_nsclc():
    raise NotImplementedError(
        "Point this at however you already load the 897,733-cell NSCLC "
        "matrix -- should return (X_raw_counts, gene_names)."
    )


# NSCLC RUNTIME WARNING, READ BEFORE RUNNING:
# SLICE itself scales fine (511s in the paper's own Table 1, at full
# scale, k=20, 3000 HVGs). cNMF does NOT scale the same way: cNMF's
# n_iter replicate NMF fits are each roughly linear in cell count, and
# your 13,353-cell Haber run already took ~210s for n_iter=20. NSCLC
# at 897,733 cells is ~67x more cells -- naive extrapolation puts a
# full-scale cNMF run at multiple hours, which isn't practical
# interactively. Recommendation: run SLICE at full NSCLC scale (for
# the cross-tissue reproducibility check, which doesn't need cNMF at
# all), but subsample NSCLC to ~50,000-100,000 cells specifically for
# the SLICE-vs-cNMF comparison. Uncomment the subsample line below.

def add_pancreas_and_nsclc(results: dict, k_pancreas: int = 15, k_nsclc: int = 20,
                            nsclc_subsample_for_cnmf: int = 80_000):
    """Extends your existing `results` dict (already has intestine,
    glioblastoma, Bcan) with pancreas and NSCLC. Returns the updated
    dict plus a separate dict of (usage, spectra_tpm, genes_cnmf) for
    the datasets you also ran cNMF on, for the agreement check."""
    cnmf_results = {}

    # --- pancreas ---
    X_panc_raw, gn_panc_raw = load_pancreas()
    Xc_panc, gn_panc = preprocess(X_panc_raw, gn_panc_raw, n_top_hvg=3000)
    results["pancreas"] = fit_slice(Xc_panc, gn_panc, k=k_pancreas, sparsity=0.02)
    dt._top_hub_gene_sets(results["pancreas"], 50)  # sanity: must not raise
    print(f"pancreas: k={k_pancreas}, {Xc_panc.shape[1]} HVGs, "
          f"{int((results['pancreas'].labels == -1).sum())} grey")

    # --- NSCLC (full scale for SLICE) ---
    X_nsclc_raw, gn_nsclc_raw = load_nsclc()
    Xc_nsclc, gn_nsclc = preprocess(X_nsclc_raw, gn_nsclc_raw, n_top_hvg=3000)
    results["nsclc"] = fit_slice(Xc_nsclc, gn_nsclc, k=k_nsclc, sparsity=0.02)
    print(f"nsclc: k={k_nsclc}, {Xc_nsclc.shape[1]} HVGs, "
          f"{int((results['nsclc'].labels == -1).sum())} grey")

    return results, cnmf_results


def run_cnmf_agreement_for(name: str, raw_h5ad_path: str, k: int, subsample_n: int = None):
    """SLICE-vs-cNMF agreement for one more dataset, matching the
    pattern from your existing cell 16-18. Pass subsample_n for NSCLC
    (see runtime warning above); leave None for pancreas/others."""
    import scanpy as sc

    if subsample_n is not None:
        adata = sc.read_h5ad(raw_h5ad_path)
        if adata.n_obs > subsample_n:
            idx = np.random.default_rng(0).choice(adata.n_obs, subsample_n, replace=False)
            adata = adata[idx].copy()
        tmp_path = raw_h5ad_path.replace(".h5ad", f"_sub{subsample_n}.h5ad")
        adata.write(tmp_path)
        raw_h5ad_path = tmp_path

    usage, spectra_tpm, genes_cnmf, top_genes, t_cnmf = cs_cnmf.run_cnmf(
        raw_h5ad_path, k=k, output_dir=f"cnmf_{name}_output", name=name, num_highvar_genes=2000,
    )
    print(f"{name} cNMF done in {t_cnmf:.1f}s")

    overdispersed_genes = open(f"cnmf_{name}_output/{name}/{name}.overdispersed_genes.txt").read().split()
    import scanpy as sc
    adata = sc.read_h5ad(raw_h5ad_path)
    adata_hvg = adata[:, adata.var_names.isin(overdispersed_genes)]
    Xlog = sp.csr_matrix(np.log1p(adata_hvg.X.toarray() if sp.issparse(adata_hvg.X) else adata_hvg.X))
    U, D, V, genes_slice, best_sparsity, t_slice = cs_cnmf.run_slice(Xlog, adata_hvg.var_names.tolist(), k=k)
    print(f"{name} SLICE done in {t_slice:.1f}s, best sparsity={best_sparsity}")

    agreement = cs_cnmf.evaluate_cross_method_agreement(V, genes_slice, spectra_tpm, genes_cnmf)
    return agreement


# =================================================================
# (2) GO enrichment on your two strongest cross-tissue findings
# =================================================================
def run_go_on_key_modules(results: dict):
    """The myeloid (glioblastoma M0 / Bcan M2) and stromal
    (glioblastoma M3 / Bcan M5) matches specifically -- excluding the
    genes you already used to identify them, so a significant hit
    here is evidence of biology beyond what you already knew to look
    for. Uses your existing discovery_toolkit.go_enrichment_beyond_markers
    -- not network-tested from my sandbox, so eyeball the first one
    before trusting the loop.
    """
    myeloid_known = ["CD74", "CTSB", "DENND3", "FTH1", "GRK2", "MAF", "PSAP",
                      "TYROBP", "C1QA", "C1QB"]
    stromal_known = ["COL16A1", "COL5A1", "COL6A1"]

    jobs = [
        ("glioblastoma_M0_myeloid", results["glioblastoma"], 0, myeloid_known),
        ("Bcan_M2_myeloid", results["Bcan"], 2, myeloid_known),
        ("glioblastoma_M3_stromal", results["glioblastoma"], 3, stromal_known),
        ("Bcan_M5_stromal", results["Bcan"], 5, stromal_known),
    ]

    out = {}
    for label, result, module_id, known in jobs:
        print(f"\n=== {label} ===")
        df_go = dt.go_enrichment_beyond_markers(result, module_id=module_id, known_marker_genes=known)
        if len(df_go):
            print(df_go[["Term", "Overlap", "P-value", "Adjusted P-value"]].head(10).to_string(index=False))
        else:
            print("(no result -- check module_id is right for this result object, or too few genes remained)")
        out[label] = df_go
    return out


# =================================================================
# (3) Flag (don't silently drop) the dissociation-artifact match
# =================================================================
# Core dissociation/stress-response gene set from the scRNA-seq QC
# literature (van den Brink et al. 2017 Nat Methods and follow-up work
# extending it) -- treat this as a reasonable starting list, not a
# guaranteed-complete one; if this matters for the paper, check it
# against the specific published list you want to cite.
DISSOCIATION_GENES = {
    "FOS", "FOSB", "JUN", "JUNB", "JUND", "ATF3", "EGR1", "EGR2", "EGR3",
    "IER2", "IER3", "IER5", "DUSP1", "ZFP36", "ZFP36L1", "ZFP36L2",
    "NR4A1", "NR4A2", "NR4A3", "KLF2", "KLF4", "KLF6", "SOCS3", "BTG2",
    "PPP1R15A", "HSPA1A", "HSPA1B", "HSPA1L", "HSPB1", "HSP90AA1",
    "HSP90AB1", "DNAJB1", "CYR61", "CCN1", "RGS1", "SAT1", "NFKBIA",
    "NFKBIZ", "GADD45B", "MCL1", "PLK2",
}


def flag_dissociation_matches(df_real_cross: pd.DataFrame, gene_col: str = "shared_genes",
                               threshold: float = 0.5) -> pd.DataFrame:
    """Adds frac_dissociation_genes and likely_dissociation_artifact
    columns to your cross-tissue table. Doesn't drop anything --
    review flagged rows yourself before deciding whether to exclude or
    footnote them."""
    def frac(genes_str):
        genes = [g.strip().upper() for g in str(genes_str).split(",") if g.strip()]
        if not genes:
            return 0.0
        return sum(1 for g in genes if g in DISSOCIATION_GENES) / len(genes)

    out = df_real_cross.copy()
    out["frac_dissociation_genes"] = out[gene_col].apply(frac)
    out["likely_dissociation_artifact"] = out["frac_dissociation_genes"] >= threshold
    return out


if __name__ == "__main__":
    print(__doc__)


# =================================================================
# HOW TO RUN THIS IN YOUR NOTEBOOK
# =================================================================
# 1) Extend to 5 datasets:
#      results, _ = add_pancreas_and_nsclc(results)   # results already
#                                                        # has intestine/
#                                                        # glioblastoma/Bcan
#      df_real_cross_5way = dt.cross_tissue_module_reproducibility(results, top_n=50)
#
# 2) cNMF agreement for pancreas (fast) and NSCLC (subsampled -- see
#    the runtime warning above):
#      agreement_pancreas = run_cnmf_agreement_for(
#          "pancreas", "pancreas_raw.h5ad", k=15)
#      agreement_nsclc = run_cnmf_agreement_for(
#          "nsclc", "nsclc_raw.h5ad", k=20, subsample_n=80_000)
#
# 3) GO enrichment on your two strongest findings:
#      go_results = run_go_on_key_modules(results)
#
# 4) Flag the dissociation match before it goes near the manuscript:
#      df_flagged = flag_dissociation_matches(df_real_cross)
#      df_flagged[df_flagged.likely_dissociation_artifact]   # inspect
#      df_clean = df_flagged[~df_flagged.likely_dissociation_artifact]  # exclude
