# =================================================================
# PASTE THESE AS NEW CELLS, in order, after your existing cell 20
# (the one that builds `results = {"intestine":..., "glioblastoma":...,
# "Bcan":...}`). Needs the updated slice-lca-package.zip re-uploaded
# and reinstalled first (it now includes benchmarks/extend_and_annotate.py).
# =================================================================


# --- CELL: download + convert NSCLC ---------------------------------
# Same 10x CellRanger .h5 -> .h5ad conversion as your glioblastoma/
# breast cancer cells (this file needs it too -- it's the same format).
nsclc_url = "https://cf.10xgenomics.com/samples/cell-exp/7.1.0/16plex_900k_32_NSCLC_multiplex/16plex_900k_32_NSCLC_multiplex_count_filtered_feature_bc_matrix.h5"
download_verified(nsclc_url, "NSCLC_900k.h5", min_expected_mb=1000)
convert_10x_h5_to_anndata("NSCLC_900k.h5", "NSCLC_900k.h5ad")


# --- CELL: fit pancreas + NSCLC, add to your existing `results` dict --
# NOTE: NSCLC's file is "filtered_feature_bc_matrix" -- 10x's own
# cell-calling already ran, so this is NOT full of empty droplets the
# way the breast cancer RAW file was. skip_qc=True here matches what
# the paper's own original pipeline did for this exact reason (same
# as glioblastoma, which is also a "filtered" file). k=20, n_top_hvg=
# 3000 match Table 1.
from slice_lca.preprocessing import preprocess
from slice_lca.core import fit_slice
import scvelo as scv
import gc

# pancreas
adata_panc = scv.datasets.pancreas()
X_panc_raw = adata_panc.X
if not sp.issparse(X_panc_raw):
    X_panc_raw = sp.csr_matrix(X_panc_raw)
gene_names_panc_raw = adata_panc.var_names.to_numpy()
del adata_panc
gc.collect()

Xc_panc, gn_panc = preprocess(X_panc_raw, gene_names_panc_raw, n_top_hvg=3000)
results["pancreas"] = fit_slice(Xc_panc, gn_panc, k=15, sparsity=0.02)
summarize_result(results["pancreas"], label="pancreas")

# NSCLC
Xraw_nsclc, gene_names_nsclc = load_counts("NSCLC_900k.h5ad")
Xc_nsclc, gn_nsclc = preprocess(Xraw_nsclc, gene_names_nsclc, n_top_hvg=3000, skip_qc=True)
results["nsclc"] = fit_slice(Xc_nsclc, gn_nsclc, k=20, sparsity=0.02)
summarize_result(results["nsclc"], label="nsclc")


# --- CELL: full 5-way cross-tissue reproducibility -------------------
import extend_and_annotate as ea

df_real_cross_5way = dt.cross_tissue_module_reproducibility(results, top_n=50)
df_real_cross_5way.head(30)


# --- CELL: flag the dissociation-artifact match before anything else -
# Run this BEFORE you eyeball the table above for real findings, so
# you're not tempted to read into a technical artifact.
df_cross_flagged = ea.flag_dissociation_matches(df_real_cross_5way)
print("flagged as likely dissociation artifact:")
print(df_cross_flagged[df_cross_flagged.likely_dissociation_artifact][
    ["dataset_A", "module_A", "dataset_B", "module_B", "jaccard", "shared_genes"]
].to_string(index=False))

df_cross_clean = df_cross_flagged[~df_cross_flagged.likely_dissociation_artifact].reset_index(drop=True)
df_cross_clean.head(20)


# --- CELL: cNMF agreement for pancreas (fast, full scale is fine) ----
agreement_pancreas = ea.run_cnmf_agreement_for("pancreas", "pancreas_raw.h5ad", k=15)
# NOTE: this needs pancreas_raw.h5ad on disk -- if you didn't already
# write one, add before this cell:
#   import scanpy as sc
#   adata_panc2 = scv.datasets.pancreas()
#   adata_panc2.write("pancreas_raw.h5ad")
agreement_pancreas


# --- CELL: cNMF agreement for NSCLC (SUBSAMPLED -- read this first) --
# Your Haber run (13,353 cells) took ~210s for cNMF. NSCLC is ~67x
# more cells -- a full-scale cNMF run here would likely take multiple
# hours. This subsamples to 80,000 cells for the cNMF side only; SLICE
# itself already ran at full scale above.
agreement_nsclc = ea.run_cnmf_agreement_for(
    "nsclc", "NSCLC_900k.h5ad", k=20, subsample_n=80_000,
)
agreement_nsclc


# --- CELL: GO enrichment on your two strongest cross-tissue findings -
# Test this cell once and eyeball the output before trusting it's
# hitting the real Enrichr API correctly (needs internet, untested
# from my end).
go_results = ea.run_go_on_key_modules(results)
