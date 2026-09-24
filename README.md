# SLICE

**S**calable **L**atent **I**nterpretable **C**omponent **E**xtraction — sparse,
signed, deflationary factorization for single-cell gene programs.

SLICE decomposes a (cells × genes) expression matrix into sparse, signed
gene programs and their per-cell activity scores, without densifying
sparse input or constructing a gene × gene correlation matrix. See the
[paper](PLACEHOLDER) for the full method and benchmarks, or
[the docs](PLACEHOLDER) for tutorials, an API reference, and honest
comparisons against cNMF, MOFA+, and WGCNA/hdWGCNA.

## Install

```bash
git clone https://github.com/Sizerta/SLICE.git
cd slice-lca
pip install -e ".[dev]"
```

## Quickstart

```python
import scipy.sparse as sp
from slice_lca import fit_slice

# X: (n_cells, n_genes) sparse, centered/log-normalized expression matrix
result = fit_slice(X, gene_names=gene_names, sparsity=0.02, min_kME=0.05)

result.U        # (n_cells, k) cell-level program activities
result.V        # (n_genes, k) signed, sparse gene loadings
result.kME      # (n_genes, k) gene-module membership correlations
result.labels   # (n_genes,) assigned module index, or -1 ("grey")
```

For n ≥ 100,000 cells, supply `k` explicitly (automatic rank selection
via parallel analysis is disabled above that scale for cost reasons):

```python
result = fit_slice(X, k=20, sparsity=0.02)
```

## Choosing hyperparameters

See `benchmarks/` for the sensitivity sweeps behind these
recommendations (10 seeds each, synthetic data with known ground truth):

- **`sparsity`**: controls the L1 budget `c = max(sqrt(sparsity * p), 1)`.
  Recovery is best when `sparsity` is set near the true fraction of
  genes you expect per program (e.g. ~50 genes out of 2,000 ⇒
  `sparsity ≈ 0.02–0.05`); values below that under-capture true hub
  genes, and — because the L1 budget saturates once
  `sqrt(sparsity·p) ≥ sqrt(p)` — values much above roughly `10×` that
  point stop adding any real sparsity control at all.
  See `benchmarks/sensitivity_sparsity_summary.csv`.
- **`min_kME`**: the module-assignment threshold. Stable across
  `[0.1, 0.5]` (100% recall, ≥99.8% precision on true module genes in
  our synthetic benchmark); `0.05` is too lenient and assigns many
  background genes. See `benchmarks/sensitivity_minkme_summary.csv`.
- **`k`**: safer to overestimate than underestimate. Extra components
  beyond the true number cause no measurable harm; underestimating k
  degrades subspace recovery even when the components that *are*
  extracted still look individually correct.
  See `benchmarks/sensitivity_k_summary.csv`.

## Biological annotation & WGCNA-style reporting

```python
from slice_lca.annotation import load_reference_db, annotate_modules, export_hub_genes
from slice_lca.bio_outputs import module_trait_table, hub_gene_table, export_hub_network

db = load_reference_db("CellMarker_2.0.xlsx", species="Human")
annotation = annotate_modules(result, db)          # BH-corrected cell-type calls, not raw p<threshold
trait_table = module_trait_table(result.U, clinical_trait)   # BH-corrected module-trait correlations
hubs = export_hub_genes(result, annotation)
```

Install the extra dependencies these need with `pip install -e ".[bio]"`.
See `CHANGELOG.md` for the specific bugs this fixed relative to the
original drafts (most importantly: module cell-type identification
now corrects for multiple testing across the reference database,
rather than comparing one best-of-many p-value to a fixed threshold).

## Real (not proxy) comparison against cNMF

`benchmarks/compare_slice_cnmf.py` runs the actual `cnmf` package
(prepare → factorize → combine → consensus), not a hand-rolled proxy,
against SLICE on the same data in the same session. Run
`python compare_slice_cnmf.py` for a self-test on synthetic
ground-truth data (already verified — see the comment at the top of
the file for the numbers it should reproduce), then see the bottom of
the file for how to point it at a real dataset. Install with
`pip install -e ".[discovery]"`.

**Honest finding from the self-test**: on data with real gene-set
overlap but no antagonistic/signed structure, cNMF's gene-loading
recovery (0.999) beat SLICE's (0.78–0.98, sparsity-dependent) even
though SLICE was ~30x faster. SLICE's real edge over cNMF isn't
"better at everything" — it's (a) representing signed/antagonistic
programs, which cNMF structurally cannot do at all (see the
antagonistic-axis benchmark), and (b) speed at scale. Don't claim more
than that without re-running this comparison on your real datasets.

## Discovery beyond marker recapitulation

`benchmarks/discovery_toolkit.py` — three ways to show SLICE finds
something beyond what you already knew to look for, in order of how
much I could verify without your real data:

1. `cross_tissue_module_reproducibility()` — does a module recur
   across independent tissues with a consistent hub-gene signature?
   Uses only datasets you already have. Fully tested (`pytest`).
2. `held_out_marker_recovery()` — do curated marker genes land
   together in an unsupervised module, and what *other* genes come
   along for free? Uses only datasets you already have. Fully tested.
3. `go_enrichment_beyond_markers()` — Enrichr pathway enrichment on a
   module's hub genes with known markers excluded. Correct against
   `gseapy`'s real API, but **not network-tested** (this sandbox can't
   reach `maayanlab.cloud`) — test it on one module yourself first.

## Preprocessing and centering (read this before fitting real data)

```python
from slice_lca.preprocessing import preprocess
from slice_lca.core import fit_slice

X_ready, gene_names_ready = preprocess(X_raw, gene_names)   # QC -> normalize+log1p -> HVG -> scale
result = fit_slice(X_ready, gene_names_ready, k=15, sparsity=0.02)  # centers implicitly by default
```

`fit_slice` centers implicitly by default (`center=True`) and raises
`NotPreprocessedError` if the input looks like raw counts. Both exist
because of a real, serious finding -- see CHANGELOG.md's "Implicit
centering" entry: the paper's own original pipeline never explicitly
centered the data, and `sparse_svd` had no implicit-centering
mechanism either, despite the Methods section describing one. This
matters most for continuous, signed/antagonistic axes (SLICE's actual
differentiating claim over NMF) -- if you have existing results
generated before this fix, rerun them.

If you have marker genes or pathway members you already know matter,
protect them from HVG filtering explicitly -- variance-based selection
can and does drop real, low-expression signal (see CHANGELOG.md's
`force_include` entry for the real case, involving two Tuft cell
markers, that motivated adding this):

```python
X_ready, gene_names_ready = preprocess(
    X_raw, gene_names, n_top_hvg=3000,
    force_include=["Dclk1", "Trpm5"],
)
```

## Robustness options for the core factorization

All default to exactly the old behavior (see
`tests/test_multistart.py` for the regression tests pinning this
down):

```python
# Reduce bad-local-optimum risk (validated to help, modestly, with a
# real held-out-seed check -- see CHANGELOG.md):
result = fit_slice(X, k=15, sparsity=0.02, n_restarts=4)

# See whether a fit actually converged, instead of it failing silently:
U, D, V, diagnostics = sparse_svd(X, k=15, sparsity=0.02, return_diagnostics=True)
print([d["converged"] for d in diagnostics])
```

## Development

```bash
pip install -e ".[dev,bio]"
pytest                 # unit + regression tests
```

The regression tests in `tests/test_recovery.py` pin down the paper's
published benchmark numbers; if you change the algorithm, re-run the
scripts in `benchmarks/` and update both the paper and these
assertions together.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE). *(Confirm this is the intended license
before publishing; MIT is a common permissive default for academic
software but the choice is yours.)*
