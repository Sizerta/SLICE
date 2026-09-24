# Quickstart

## Install

```bash
pip install slice-lca          # once published to PyPI
```

For development, or before that PyPI release exists:

```bash
git clone PLACEHOLDER
cd slice-lca
pip install -e ".[dev,bio]"
```

## The five-minute version

`fit_slice` expects data that has already been through QC, library-size
normalization, `log1p`, HVG selection, and variance scaling — not raw
counts. If you skip this, `fit_slice` will raise `NotPreprocessedError`
rather than silently produce garbage (this used to happen silently; see
[Preprocessing](preprocessing.md) for why that was worth fixing).

```python
from slice_lca.preprocessing import preprocess
from slice_lca.core import fit_slice

# X_raw: (n_cells, n_genes) sparse or dense raw count matrix
X_ready, gene_names_ready = preprocess(X_raw, gene_names)

result = fit_slice(X_ready, gene_names_ready, k=15, sparsity=0.02)

result.U        # (n_cells, k) cell-level program activities
result.V        # (n_genes, k) signed, sparse gene loadings
result.kME      # (n_genes, k) gene-module membership correlations
result.labels   # (n_genes,) assigned module index, or -1 ("grey" / unassigned)
```

## Reading the output

`result.V[:, m]` is module `m`'s gene loadings. Unlike NMF's `W`, these
can be negative — genes with opposite loading sign on the same
component are moving in opposite directions along the same latent
axis, not necessarily regulating each other directly (see
[Preprocessing](preprocessing.md#what-a-signed-loading-does-and-does-not-mean)
for exactly what this claim does and doesn't support).

To see which genes drive a module, sorted by absolute loading strength:

```python
import numpy as np

module_id = 1
top_n = 15
order = np.argsort(-np.abs(result.kME[:, module_id]))[:top_n]
for i in order:
    print(gene_names_ready[i], result.kME[i, module_id])
```

Genes with positive `kME` on this module load the same direction as
its positive pole; negative `kME` genes load the opposite direction.
Both are real hub genes of the same signed component — this is the
whole point of not being forced into NMF's non-negativity constraint.

## Choosing `k`

For datasets under 100,000 cells, leave `k` unset and SLICE will pick
it via parallel analysis against column-permuted null matrices:

```python
result = fit_slice(X_ready, gene_names_ready, sparsity=0.02)  # k chosen automatically
```

Above that, parallel analysis gets expensive enough that `fit_slice`
requires `k` explicitly rather than silently taking a long time:

```python
result = fit_slice(X_ready, gene_names_ready, k=20, sparsity=0.02)
```

If you're unsure what `k` should be, the paper's own hyperparameter
sensitivity results are relevant here: overestimating `k` costs very
little (subspace recovery is flat once `k` reaches the true rank),
while underestimating it measurably degrades recovery even for the
components that do get extracted. When in doubt, go a bit higher
rather than lower.

## Choosing `sparsity`

Recovery is best when `sparsity` is set near the true fraction of
genes you actually expect in a program — for roughly 50 genes out of
2,000, that's `sparsity ≈ 0.02–0.05`. Values well above the true
fraction stop adding meaningful sparsity control once the ℓ1 budget
saturates; values well below it under-capture real hub genes. The
`benchmarks/` sensitivity sweeps behind this recommendation (10 seeds
each, synthetic ground truth) are in the repository if you want the
full curves rather than the summary.

## If a fit looks unstable

SLICE initializes each component with a random vector, so different
seeds can converge to different (though usually correlated) solutions
— more so under heavy cross-program gene overlap. If you're not
confident a single fit is representative:

```python
from slice_lca.core import sparse_svd_multistart

U, D, V, diagnostics, spread = sparse_svd_multistart(
    X_ready, k=15, n_restarts=4, sparsity=0.02
)
```

This costs `n_restarts` times the runtime of a single fit. It's worth
it when program overlap is heavy; it's unnecessary overhead when it
isn't.
