# Preprocessing

Read this before running SLICE on real data. Two real failures in this
project's history came directly from skipping or misunderstanding this
step, and both are worth understanding rather than just working around.

## The pipeline

```python
from slice_lca.preprocessing import preprocess

X_ready, gene_names_ready = preprocess(X_raw, gene_names, n_top_hvg=3000)
```

This runs, in order: QC filtering (`qc_filter`), library-size
normalization + `log1p` (`normalize_log1p`), highly-variable-gene
selection (`select_hvg`), and per-gene variance scaling
(`scale_only`). It deliberately does **not** explicitly mean-center
the result — `fit_slice` centers implicitly by default, and calling
`center_explicit()` yourself first would densify the matrix, exactly
the memory cost this whole design exists to avoid.

## Failure 1: calling `fit_slice` on raw counts directly

Before `preprocessing.py` existed as a module, the steps above only
lived as ad hoc code pasted into individual analysis notebooks. A real
analysis call skipped straight to `fit_slice()` on raw, unnormalized
counts — no QC, no `log1p`, no HVG restriction, no variance scaling —
and got back "modules" claiming 10,000+ genes each. Not subtly wrong;
obviously degenerate.

`fit_slice` now checks for this directly. If more than about 95% of a
sample of the input's nonzero values are exact integers, it raises
`NotPreprocessedError` instead of running:

```python
fit_slice(X_raw, gene_names)
# NotPreprocessedError: Input looks like raw, unnormalized counts...
```

If you're intentionally passing something unusual (already-normalized
data that happens to look integer-like, say), `validate_input=False`
turns this check off. For anything else, run `preprocess()` first.

## Failure 2: centering, and why it's not optional

This is the more serious one. The paper's Methods section describes
implicit centering during the sparse power iteration. For a long time,
no such mechanism actually existed in the code — every real-dataset
result reported at some point (marker-gene `kME` values, Table 1's
numbers) came from a pipeline that normalizes, log1p's, and
variance-scales, but never centers.

On a synthetic axis with known up- and down-regulated genes, disabling
the centering step drops gene-loading recovery from **0.958 ± 0.004**
to **0.003 ± 0.002** (8 seeds, identical input otherwise) —
indistinguishable from chance. An uncentered fit is dominated by
overall expression level rather than genuine signed covariation: the
same failure mode that motivates centering in ordinary PCA, just
easier to miss here because the matrix stays sparse and nothing
obviously crashes.

`fit_slice` centers implicitly by default now (`center=True`):

```python
result = fit_slice(X_ready, gene_names_ready, k=15, sparsity=0.02)
# center=True by default -- this is the corrected behavior
```

Pass `center=False` only if you specifically need to reproduce
pre-fix numbers from before this was found.

## HVG selection can drop the genes you actually care about

Variance-based HVG selection keeps the genes with the most variance
across cells — which is not the same thing as keeping the genes that
matter most biologically. Transcription factors and other regulatory
genes are often expressed at low, tightly-buffered levels precisely
*because* their protein products are potent, and a low-variance gene
can fail the HVG cut even while sitting at the center of the
regulatory network a program actually reflects.

This happened in practice: `Dclk1` and `Trpm5` (Tuft cell markers,
independently published at `kME` ≈ 0.91) were HVG-filtered out of a
real intestinal analysis entirely, and the resulting component came
back at `kME` < 0.15 for genes that should have been strong markers —
not because the biology wasn't there, but because the relevant genes
never made it into the matrix `fit_slice` saw.

If you have genes you already know matter — a marker panel, pathway
members, anything you can't afford to lose to a variance cutoff —
keep them explicitly:

```python
X_ready, gene_names_ready = preprocess(
    X_raw, gene_names, n_top_hvg=3000,
    force_include=["Dclk1", "Trpm5", "Lyz1"],
)
```

Matching is case-insensitive, and a name not present in your dataset
is silently ignored rather than raising an error — a marker list that
includes one gene absent from this particular tissue shouldn't cost
you the rest of the HVG selection. This does not fix the underlying
bias on its own; deciding *which* genes are worth protecting this way
still requires knowing your biology going in. See the paper's
Limitations (§4.3) for the fuller discussion.

## What a signed loading does and does not mean

Once you have a fitted result, it's worth being precise about what
`result.V[:, m]`'s sign actually tells you. Genes with opposite
loading sign on the same component are opposing transcriptional
variation axes — cells occupying different positions along that
component's latent continuum show these gene sets moving in opposite
directions. That is a statistical claim about covariation, not a
mechanistic one: it is not, by itself, evidence of repression,
inhibition, or direct regulatory antagonism between the genes
involved, and it does not distinguish between two genes trading off
*within* the same cells versus simply being expressed in different
cell subpopulations. Either would produce the same signed pattern.
Establishing which of these is actually happening needs orthogonal
evidence — perturbation, ATAC, protein-level data — that expression
covariation alone can't provide.
