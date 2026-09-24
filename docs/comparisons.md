# Comparisons against other methods

This page states plainly where SLICE wins, where it doesn't, and where
the honest answer is "it depends." All numbers here come from the
paper's real-data benchmarks (same hardware, same session), not
synthetic self-tests, unless stated otherwise.

## vs. cNMF

On the same real intestinal and pancreatic datasets, SLICE ran
8–13× faster. Cross-method agreement (Hungarian-matched components,
gene-loading correlation) was real but only moderate — mean
`|r|` of 0.30–0.35, best match up to 0.68 — meaningfully weaker than
the near-perfect agreement (`r=0.99`) both methods reach on a
synthetic self-test with matched structure. The two methods frequently
converge on different decompositions of the same real tissue.

The structural finding is the more important one: SLICE's module 1 in
the intestinal atlas represents an oxidative-phosphorylation program
and Paneth-cell secretory identity as one signed axis. Run the real
`cnmf` package on the same data, and the same genes split across four
*disjoint* components with no shared component between the two poles
— not because cNMF's implementation is worse, but because
non-negativity makes representing this any other way mathematically
impossible. See `benchmarks/compare_slice_cnmf.py` and
[Reproducing the paper](reproducing-paper.md) to run this yourself.

**Where SLICE does not have an advantage**: on data with real
gene-set overlap but no antagonistic/signed structure, cNMF's
gene-loading recovery in a synthetic self-test (0.999) beat SLICE's
(0.78–0.98, sparsity-dependent), even though SLICE was faster. If your
question doesn't involve opposing gene sets, don't expect SLICE to be
the better choice just because it's faster.

## vs. MOFA+

MOFA+'s spike-and-slab and automatic-relevance-determination priors
permit signed loadings — it is not structurally blocked from
representing an antagonistic axis the way cNMF is, and on the same
real intestinal data it *did* recover module 1 correctly, as a single
factor with the same genes carrying opposing signs SLICE found. This
is worth being direct about: MOFA+ can do this task.

Where the methods diverge is computational cost. MOFA+ requires
densifying sparse input (it does not accept sparse matrices directly)
and took roughly 30× longer than SLICE on identical data in this
comparison (860–877s vs. 28.2s, same 3,000-gene panel, same hardware).
A reliable peak-memory comparison could not be obtained in the
environment this was measured in — repeated measurements returned an
identical value regardless of which method or dataset size was run,
indicating the process-level peak was being set by something other
than either method's own fit. That number is reported as unmeasured
rather than guessed at.

## vs. WGCNA / hdWGCNA

Classical WGCNA's pairwise gene-correlation matrix is `O(p²)` in gene
count — not feasible at single-cell scale regardless of implementation
quality. hdWGCNA addresses this by aggregating cells into metacells
before network construction, which is a real, principled solution to
single-cell dropout, not merely a workaround — but it means the method
operates on averaged pseudo-samples rather than individual cells, and
in its own published scalability demonstration (Morabito et al. 2023,
~965k PBMCs), reaching that scale required both metacell aggregation
and manually partitioning the data into five separate per-cell-type
analyses run independently. SLICE was not run head-to-head against
hdWGCNA on the same hardware in this project — the current release
could not be installed in the environment used for the other
comparisons — so this section states the structural difference rather
than a controlled runtime number.

## vs. PCA / Sparse PCA / ICA / GLM-PCA

PCA and ICA both permit signed loadings; neither is sparse in its
standard form, so individual components don't isolate a small,
interpretable gene set the way SLICE's does. Sparse PCA adds sparsity
but, in synthetic benchmarks under severe gene overlap (0.8 overlap
fraction), fragments the latent subspace (subspace error 0.412 ±
0.035) where SLICE's deflation mechanism stays close to PCA's
theoretical optimum (0.035 ± 0.001). GLM-PCA models counts through an
explicit generalized-linear likelihood — a genuinely different and
sometimes more appropriate approach to normalization — but is not
sparse in the loading sense and fits via iterative reweighting rather
than a small number of sparse matrix-vector products.

## When SLICE is probably not the right tool

- You need non-linear structure (trajectories, complex manifolds) —
  look at scVI or a similar deep generative approach instead.
- You need causal or mechanistic claims about the genes you find —
  SLICE (like every method on this page) gives you statistical
  association, not evidence of regulation. See
  [Preprocessing](preprocessing.md#what-a-signed-loading-does-and-does-not-mean).
- Your data has no reason to contain antagonistic/opposing gene
  programs — cNMF's synthetic self-test recovery above suggests it
  may do at least as well, faster to set up if you already have it in
  your pipeline.
