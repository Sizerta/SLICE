"""
Two things, both operating on your already-fitted `results` dict
(the one built in your notebook: {"pancreas":..., "intestine":...,
"glioblastoma":..., "Bcan":..., "nsclc":...}, using the CURRENT,
centered pipeline):

  (1) reverify_markers()      -- direct old-vs-new comparison for every
                                  marker-gene kME value currently published
                                  in the paper. This is the single most
                                  important open question right now: were
                                  these numbers generated before or after
                                  the centering fix?
  (2) find_antagonistic_modules() -- systematically scans EVERY module in
                                  EVERY dataset for genuine bidirectional
                                  (signed) structure, ranked by strength.
                                  This directly answers the reviewer
                                  critique asking for "one flagship
                                  biological example where the signed
                                  nature actually matters" -- found by
                                  scanning, not by picking the one that
                                  looks best after the fact.
"""
import numpy as np
import pandas as pd


# =================================================================
# (1) Reverify every published marker-gene kME value
# =================================================================
# Exactly what's currently in the manuscript -- the numbers this
# function will compare your NEW (centered-pipeline) results against.
PUBLISHED_MARKERS = {
    "pancreas": {
        "NEUROG3": 0.84,   # endocrine progenitor
        "SOX9": -0.70,     # ductal (claimed antagonistic to NEUROG3)
    },
    "intestine": {
        "MUC2": 0.83, "TFF3": 0.83,     # Goblet
        "LYZ1": -0.70,                    # Paneth
        "DCLK1": -0.91, "TRPM5": -0.91,   # Tuft
    },
    "Bcan": {
        "PTPRC": None, "CD3E": None,   # M1 immune module -- paper reports
                                          # module identity, not a specific
                                          # kME number, so this checks
                                          # which module they land in now
    },
}


def reverify_markers(results: dict) -> pd.DataFrame:
    """For every gene in PUBLISHED_MARKERS, find its current module
    and kME in `results`, and compare against the published value.
    Flags anything that changed sign (the specific failure mode the
    centering bug would cause) or moved by more than 0.15 in
    magnitude (a threshold, not a hard rule -- eyeball the smaller
    shifts too)."""
    rows = []
    for dataset, markers in PUBLISHED_MARKERS.items():
        if dataset not in results:
            continue
        result = results[dataset]
        gene_to_idx = {g.upper(): i for i, g in enumerate(result.gene_names)}
        for gene, published_kme in markers.items():
            idx = gene_to_idx.get(gene.upper())
            if idx is None:
                rows.append({"dataset": dataset, "gene": gene, "published_kME": published_kme,
                             "current_module": "NOT FOUND", "current_kME": np.nan,
                             "sign_flipped": None, "shift": np.nan})
                continue
            module = int(result.labels[idx])
            current_kme = float(result.kME[idx, module]) if module >= 0 else float(np.max(np.abs(result.kME[idx])) * np.sign(result.kME[idx][np.argmax(np.abs(result.kME[idx]))]))
            sign_flipped = None
            shift = np.nan
            if published_kme is not None:
                sign_flipped = (np.sign(current_kme) != np.sign(published_kme))
                shift = abs(current_kme - published_kme)
            rows.append({
                "dataset": dataset, "gene": gene, "published_kME": published_kme,
                "current_module": f"M{module}" if module >= 0 else "grey",
                "current_kME": round(current_kme, 3),
                "sign_flipped": sign_flipped,
                "shift": round(shift, 3) if not np.isnan(shift) else np.nan,
            })
    df = pd.DataFrame(rows)
    n_flipped = df["sign_flipped"].sum() if df["sign_flipped"].notna().any() else 0
    print(f"=== {len(df)} marker genes checked, {int(n_flipped)} sign flips ===")
    if n_flipped > 0:
        print("SIGN FLIPS (the specific failure mode the centering bug would cause):")
        print(df[df.sign_flipped == True].to_string(index=False))
    print(df.to_string(index=False))
    return df


# =================================================================
# (2) Find genuinely antagonistic modules, systematically
# =================================================================
def find_antagonistic_modules(results: dict, top_n_hub: int = 30, min_module_size: int = 15) -> pd.DataFrame:
    """For every module in every dataset, compute what fraction of its
    top hub genes carry NEGATIVE kME vs positive -- a module near 50/50
    is a genuine single-axis antagonistic program (SLICE's actual
    differentiator); a module near 0% or 100% negative is just a
    regular one-directional module (the kind any method could find).
    Ranks candidates by closeness to 50/50, restricted to modules big
    enough that the split isn't noise.

    This is the systematic version of "find one flagship signed
    example" -- scan everything, rank by how antagonistic it genuinely
    is, then go read the actual genes for the top few candidates
    before claiming any of them as a headline result.
    """
    rows = []
    for dataset, result in results.items():
        for m in range(result.k):
            idx = np.where(result.labels == m)[0]
            if len(idx) < min_module_size:
                continue
            kme_vals = result.kME[idx, m]
            order = idx[np.argsort(-np.abs(kme_vals))][:top_n_hub]
            hub_kme = result.kME[order, m]
            n_pos = int((hub_kme > 0).sum())
            n_neg = int((hub_kme < 0).sum())
            frac_neg = n_neg / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0.0
            balance = 1 - abs(frac_neg - 0.5) * 2  # 1.0 = perfect 50/50, 0.0 = all one direction
            top_pos = [result.gene_names[i] for i in order[hub_kme > 0][:6]]
            top_neg = [result.gene_names[i] for i in order[hub_kme < 0][:6]]
            rows.append({
                "dataset": dataset, "module": f"M{m}", "module_size": len(idx),
                "n_pos_hub": n_pos, "n_neg_hub": n_neg, "balance_score": round(balance, 3),
                "top_positive_genes": ", ".join(top_pos), "top_negative_genes": ", ".join(top_neg),
            })
    df = pd.DataFrame(rows).sort_values("balance_score", ascending=False).reset_index(drop=True)
    print(f"=== scanned {len(df)} modules across {len(results)} datasets ===")
    print(f"top 10 candidates by antagonistic balance (closest to a real 50/50 signed axis):")
    print(df.head(10).to_string(index=False))
    return df


if __name__ == "__main__":
    print(__doc__)


def inspect_module(result, module_id: int, top_n: int = 20):
    """Quick lookup: what does a specific module actually contain?
    Use this on pancreas M1 to see what SOX9's new home represents."""
    idx = np.where(result.labels == module_id)[0]
    kme_vals = result.kME[idx, module_id]
    order = idx[np.argsort(-np.abs(kme_vals))][:top_n]
    df = pd.DataFrame({
        "gene": [result.gene_names[i] for i in order],
        "kME": [round(float(result.kME[i, module_id]), 3) for i in order],
    })
    print(f"module M{module_id}: {len(idx)} genes total, top {top_n} by |kME|:")
    print(df.to_string(index=False))
    return df


def go_enrich_antagonistic_module(result, module_id: int, known_genes: list):
    """GO enrichment on a candidate antagonistic module, excluding genes
    you've already identified by name (e.g. LYZ1 + the obvious OXPHOS/
    defensin hits) -- a significant result here means there's more real
    biology in this module than what's already visually obvious."""
    import discovery_toolkit as dt
    return dt.go_enrichment_beyond_markers(result, module_id=module_id, known_marker_genes=known_genes)


def check_gene_survives_hvg(gene_names_full: list, gene_names_hvg: list, genes_of_interest: list):
    """Before concluding a 'NOT FOUND' marker gene is a real problem,
    check whether it's absent from the FULL preprocessed panel (a real
    data/naming issue) or just dropped by HVG selection specifically
    (fixable by raising n_top_hvg or force-including known markers)."""
    full_upper = {g.upper() for g in gene_names_full}
    hvg_upper = {g.upper() for g in gene_names_hvg}
    rows = []
    for g in genes_of_interest:
        gu = g.upper()
        rows.append({
            "gene": g,
            "in_full_panel": gu in full_upper,
            "in_hvg_panel": gu in hvg_upper,
            "diagnosis": ("dropped by HVG selection -- raise n_top_hvg or force-include" if gu in full_upper and gu not in hvg_upper
                          else "present, should have been found -- check spelling/case" if gu in hvg_upper
                          else "absent from full panel -- check gene symbol / species annotation"),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df
