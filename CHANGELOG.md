# Changelog

## Unreleased

### `preprocessing.py`

- **Fixed**: `force_include` — cited in the paper's Limitations section
  (Section 4.3, on HVG selection bias against low-expression regulatory
  genes) as an existing mitigation, but not actually implemented
  anywhere in `select_hvg` or `preprocess` in this package. Added,
  matching the original design: case-insensitive symbol matching,
  silently ignores names absent from the dataset (does not error), no
  duplicate columns when a forced gene is already in the top-N, and
  gene names stored as `dtype=object` to avoid a real, previously-caught
  bug where a short-inferred numpy string dtype silently truncates a
  longer forced-include gene name assigned into it later.
- **Fixed**: no dedicated test file existed for this module at all
  (`test_centering.py` tests the *centering* mechanism in `core.py`, a
  different thing). Added `tests/test_preprocessing.py` — 15 tests
  covering `qc_filter`, `normalize_log1p`, `select_hvg` (including
  `force_include` specifically), `scale_only`, and `preprocess`
  end-to-end.

### `annotation.py` (module cell-type identification)

- **Fixed**: `annotate_modules` compared a single best-of-many p-value
  against a fixed, uncorrected threshold. Against a realistic ~120
  cell-type reference database, this gives an empirically-measured
  ~5% chance *per module* of confidently naming a module that has no
  real enrichment at all — see
  `tests/test_annotation.py::test_no_false_positive_identification_on_noise`,
  which reproduces this on synthetic noise. Now applies
  Benjamini-Hochberg correction across every (module, cell type) test
  in the call and reports a q-value.
- **Fixed**: `load_reference_db` grouped cell types on the raw string,
  so `"T cell"` / `"T Cell"` / `" T cell "` became three separate,
  smaller reference sets instead of one. Now normalizes on a
  lowercased/stripped key before grouping.
- **Fixed**: column auto-detection (`find_col`) picked a column
  silently; if it picked the wrong one due to a naming collision,
  there was no way to notice short of the annotations looking wrong
  downstream. Now prints the selected column for every role.
- **Fixed**: requesting species filtering when no species column
  exists silently skipped the filter. Now raises a `UserWarning`.
- **Fixed**: `plot_module_marker_heatmap` called `list.index()` inside
  a nested loop over (cell types × marker genes) — O(n_genes) per
  lookup, dominant cost on a real ~20,000-gene panel. Now a
  precomputed `gene -> index` dict.
- **Fixed**: `export_hub_genes` raised a bare `IndexError` if a
  module was missing from `annotation_df` (e.g. a stale/mismatched
  annotation table). Now warns and fills in `"Unknown"`.

### `bio_outputs.py`

- **Fixed**: `from lca_core import bh_fdr` pointed at a module that
  doesn't exist in this package — a hard `ImportError`, so the file
  could not be imported at all. `bh_fdr` now lives in
  `slice_lca.core` (tested against `statsmodels` to floating-point
  precision — `tests/test_core.py`).
- **Fixed**: `export_hub_network` passed a scipy sparse slice
  straight to `np.corrcoef`, which doesn't accept sparse input and
  raised `AttributeError` — meaning this function could not run at
  all on the sparse matrices SLICE is built around. Now densifies
  only the small `(n_cells × top_n)` submatrix it actually needs.
- **Fixed**: `plot_module_trait_heatmap` sorted module labels as
  strings (`"M10"` before `"M2"`), which is silently wrong for any
  `k >= 10` — including the paper's own NSCLC run (`k=20`). Now sorts
  numerically.
- **Fixed**: `gene_module_profile` broke on a numpy-array
  `gene_names` (`.index()` isn't defined on arrays) and required an
  exact-case match despite the rest of the pipeline upper-casing
  everything. Now accepts list or array input, matches
  case-insensitively, and raises a clear error naming the missing
  gene.
- **Documented**: `variance_explained`'s per-component ratios are not
  guaranteed to sum to the joint reconstruction's total variance,
  because SLICE's components are not enforced to be exactly
  orthogonal. This is the same caveat WGCNA-style "% variance per
  module" reporting carries; it's now stated explicitly in the
  docstring rather than left implicit.

## Colab notebook / data download

- **Fixed**: the notebook's `download_verified()` used `subprocess`
  without importing it (only worked in a live session due to leftover
  kernel state from an edited-out cell -- a fresh restart would hit
  `NameError` on the first real download).
- **Fixed**: the same function's completeness check was `size >= 1 MB`,
  which a genuinely truncated download clears easily -- this is
  exactly what happened with the Haber Zenodo dataset (stopped at
  117 MB of an expected 180 MB) and caused a confusing `OSError` three
  cells later instead of failing where the problem actually was.
  `slice_lca.data_download.download_verified()` now compares against
  the server's real `Content-Length`, retries, and re-verifies even a
  file that's already cached on disk instead of trusting its presence.
  See `tests/test_download.py`, which reproduces this exact failure
  mode (byte-for-byte truncated transfer, and a real corrupted HDF5
  file that h5py is asked to open) rather than just checking sizes.
- **Fixed**: 10x Genomics CellRanger `.h5` exports (e.g. the Breast
  Cancer and Glioblastoma Flex downloads) are a different schema from
  AnnData's `.h5ad` -- neither `sc.read_h5ad()` nor cnmf's own
  `prepare()` can read them directly (cnmf's dispatch only recognizes
  `.h5ad`/`.mtx`/`.mtx.gz`/text, so a bare `.h5` silently falls
  through to being parsed as text). Added
  `convert_10x_h5_to_anndata()`, tested against a real
  CellRanger-schema HDF5 file, not just a mock.

## Mathematical core (`sparse_svd`)

Reviewed the deflation math and rank-selection permutation logic
carefully first — both check out exactly (deflation verified
numerically equivalent to explicit deflation of X to machine
precision; the sparse per-column permutation in
`select_k_parallel_analysis` is the correct efficient equivalent of a
dense column shuffle). Nothing there needed fixing. Three new, purely
**opt-in** additions (every default is unchanged, and
`tests/test_multistart.py::test_default_signature_unchanged_returns_three_values`
plus `test_check_v_convergence_default_off_matches_published_numbers`
pin this down):

- **`sparse_svd_multistart()`** (new function) — runs `sparse_svd`
  from several random seeds and keeps the highest-explained-variance
  result. Empirically validated, not just theorized: on a
  deliberately hard synthetic regime (8 programs, 0.6 overlap, SNR
  1.5), 4 restarts improved mean gene-loading recovery correlation
  from 0.60 to 0.61, worst-case (min over trials) from 0.57 to 0.59,
  and cut the standard deviation of subspace-error by roughly a
  third — and the effect **replicated on a held-out, disjoint seed
  range** run afterward specifically to rule out it being an artifact
  of the seeds used to discover it (see
  `benchmarks/mathcore_multistart_validation.py`). `fit_slice` gained
  an `n_restarts` parameter (default 1, unchanged) that uses this
  under the hood.
- **`return_diagnostics=True`** (new `sparse_svd` parameter) —
  surfaces per-component `n_iter_used` / `converged` /
  `final_delta_d`. Right now nothing in this codebase can tell you
  whether a real fit actually converged or silently hit the
  iteration cap; this makes that visible without changing any
  results.
- **`check_v_convergence=True`** (new `sparse_svd` parameter) —
  additionally requires the gene-loading vector itself to stabilize,
  not just the scalar objective, before declaring convergence
  (strictly stricter, never loosens the old criterion).

**One negative result, reported because it's true, not because it's
convenient**: also implemented and tested `n_warmup_iter` (an
unconstrained power-iteration warm-start before switching on the
sparsity projection), on the theory that starting near the true
dominant direction should reduce bad-local-optimum outcomes. Tested
across three regimes (moderate overlap, severe overlap, and the same
hard low-SNR regime multistart was validated on) over 25-50 seeds
each. **It did not measurably help in any of them.** Left in the
code (harmless when off, essentially free when on) but explicitly
documented as unvalidated rather than presented as a fix — see its
docstring in `core.py` for the likely reason (the unconstrained
problem's landscape isn't a good proxy for the L1-constrained one).

## Implicit centering (the most significant fix in this project)

**Found from a real Colab run producing degenerate results (11,000+
gene "modules"), traced back to something much bigger than that one
notebook.** Hard evidence from the paper's own original notebook:
every real dataset in Table 1 -- including the marker-gene biology
(NEUROG3, SOX9, MUC2, etc.) -- went through `run_lca_pipeline ->
fit_lca`, which does QC + normalize + log1p + HVG + variance-scale but
**never explicitly mean-centers the data**, despite the paper's
Methods section stating "column means are incorporated implicitly
during matrix-vector multiplication." No such mechanism existed
anywhere in `sparse_svd`.

**Measured cost**: on a clean synthetic antagonistic (signed) axis,
processed exactly like the real pipeline (normalized, log1p'd,
variance-scaled, not centered), gene-loading recovery correlation with
ground truth was **0.006** -- a nearly complete failure. With
centering, **0.95**, on the identical input.

**Important nuance**: severity depends heavily on the structure being
sought. Continuous, *balanced* antagonistic axes (SLICE's actual
differentiating claim over NMF) fail almost completely without
centering. Simple discrete cell-type marker clusters degrade more
mildly (~0.85 -> ~0.73 in one test) because the sparsity constraint
partially compensates. This most threatens exactly the results that
make SLICE's core pitch, not uniformly everything in the paper.

**The fix**: `sparse_svd(..., center=True)` implements real implicit
centering -- `(X - 1*mu^T) @ v = X@v - (mu.v)*1`, computed without
ever forming the dense centered matrix (mathematically exact, verified
to machine precision against explicit centering, and confirmed to
reproduce the paper's own published ablation numbers bit-for-bit when
fed raw uncentered input -- see `tests/test_centering.py`). Confirmed
memory-bounded at scale (392 MB peak for a 200,000 x 5,000 matrix that
would need 4 GB dense). `sparse_svd`'s own default stays `center=False`
(exact backward compatibility); `fit_slice` -- the recommended entry
point -- now defaults to `center=True`.

**Also added**: `slice_lca.preprocessing` (QC/normalize/log1p/HVG/scale
-- this module didn't exist before; its absence is exactly what let
raw counts get fed straight into `fit_slice` in the first place), and
`fit_slice` now raises `NotPreprocessedError` on input that looks like
raw counts (>95% of sampled values suspiciously close to integers)
instead of silently returning degenerate results.

**Validated on the actual failure case**: re-ran the real notebook's
cross-tissue reproducibility cell (same code, realistic synthetic
scale) with the fix. Worst-offending module dropped from ~28% of the
(unfiltered) transcriptome to 11-16% of a properly HVG-restricted
panel, with normal, interpretable per-module size distributions
throughout instead of a handful of giant catch-all modules.

**Recommendation, stated plainly**: the paper's existing marker-gene
kME numbers (SOX9, LYZ1, DCLK1, etc.) were generated without this fix.
Real biological data has more redundant structure than a clean
synthetic test, so these numbers are not necessarily wrong -- but they
are unverified against the corrected pipeline. Rerun the real-dataset
analyses with `fit_slice(..., center=True)` (now the default) before
treating the current signed/antagonistic findings as final.

## cNMF comparison script: two more bugs, found from the same real run

- `evaluate_cross_method_agreement` crashed (`ValueError`, dimension
  mismatch: 2000 vs 27998) because cNMF's `load_results()` returns
  `spectra_tpm` over ALL genes in the input file (it refits its K
  programs back onto every gene via TPM -- a real, intentional cNMF
  feature, not a bug in cnmf), not just the HVG subset used for
  fitting. Fixed by tracking gene names through `run_slice`/`run_cnmf`
  and aligning by identity, not array position.
- `run_slice` previously told the caller to build
  `Xc = sp.csr_matrix(Xlog - Xlog.mean(0))` by hand, silently
  densifying. Now uses `fit_slice(..., center=True)` directly.
