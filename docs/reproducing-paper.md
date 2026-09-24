# Reproducing the paper

The scripts in `benchmarks/` fall into two genuinely different
categories, and it's worth knowing which is which before you run
anything.

## Runs standalone, right now, from this repository

These use synthetic data generated with known ground truth
(`slice_lca.datasets`) or numbers already computed and embedded in the
script itself — no external download, no real dataset required.

| Script | What it reproduces |
|---|---|
| `hyperparam_sensitivity.py` | The sparsity/`k`/`min_kME` sensitivity sweeps behind the paper's Figure 3 and the recommendations in [Quickstart](quickstart.md#choosing-sparsity) |
| `sensitivity_minkme.py` | The module-assignment threshold curve specifically (10 seeds, synthetic ground truth) |
| `mathcore_multistart_validation.py` | Whether `sparse_svd_multistart`'s best-of-*N* actually helps, checked on held-out seeds not used to tune it |
| `multiseed_antagonistic.py`, `multiseed_overlap.py` | Recovery under varying synthetic gene-overlap severity (the paper's overlap-tradeoff and antagonistic-axis figures) |
| `bh_correction.py` | Benjamini-Hochberg correction applied to the TCGA 15-module survival scan, using the already-computed p-values from that analysis |

Run any of these directly:

```bash
pip install -e ".[dev]"
python benchmarks/hyperparam_sensitivity.py
```

## Needs real data you provide yourself

These scripts implement the comparison logic and are tested against
synthetic data with a known correct answer, but the paper's actual
numbers (real intestine/pancreas/breast-cancer results) came from
public single-cell atlases that are not bundled in this repository —
for licensing and size reasons, and because re-downloading from the
original source is the more reproducible choice than a repackaged
copy.

**`compare_slice_cnmf.py`** — runs the real `cnmf` package (not a
hand-rolled stand-in) against SLICE on the same data in the same
session. The self-test at the top of the file runs on synthetic
ground truth and is already verified (see the comment there for the
exact numbers it should reproduce). Pointing it at a real dataset
needs an `.h5ad` file of raw counts; see the bottom of the file for
the exact call. Install with `pip install -e ".[discovery]"` first —
this pulls in `cnmf` and `scanpy`, which the core package doesn't
depend on.

**`discovery_toolkit.py`** — three functions, in order of how directly
testable they were without real data:

1. `cross_tissue_module_reproducibility()` — fully tested (`pytest`),
   works on any set of fitted results you already have.
2. `held_out_marker_recovery()` — fully tested, same as above.
3. `go_enrichment_beyond_markers()` — correct against `gseapy`'s real
   API, but not network-tested in the environment this was built in
   (no route to `maayanlab.cloud` from there). Test it on one module
   of your own data before trusting it on all of them.

## What isn't in this repository at all

The clinical validation (TCGA, METABRIC, cBioPortal), the DepMap
co-dependency check, and the real-data MOFA+ comparison were run
against external cohorts and data releases too large and too
frequently updated to vendor a copy of here. The paper's Methods
section documents the exact data releases and access methods used;
treat this repository as the algorithm and the synthetic-data
validation, not a full re-run of every real-data result on demand.
