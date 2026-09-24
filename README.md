# SLICE

**S**calable **L**atent **I**nterpretable **C**omponent **E**xtraction — sparse, signed, deflationary factorization for single-cell gene programs.

[![CI](https://github.com/Sizerta/SLICE/actions/workflows/CI.yml/badge.svg)](https://github.com/Sizerta/SLICE/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://sizerta.github.io/SLICE/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Sizerta/SLICE/blob/main/LICENSE)

SLICE decomposes a **cells × genes** expression matrix into sparse, signed gene programs and their per-cell activity scores.

The method is designed for sparse single-cell data and avoids both densifying the input matrix and constructing a gene × gene correlation matrix.

The [documentation](https://sizerta.github.io/SLICE/) has tutorials, an API reference, preprocessing guidance, benchmarks, and comparisons with cNMF, MOFA+, and WGCNA/hdWGCNA.

The paper and full benchmark results are linked from the project documentation.

## Installation

```bash
git clone https://github.com/Sizerta/SLICE.git
cd SLICE
pip install -e ".[dev]"
```

## Quickstart

```python
from slice_lca import fit_slice

# X: (n_cells, n_genes) sparse, centered/log-normalized expression matrix
result = fit_slice(
    X,
    gene_names=gene_names,
    sparsity=0.02,
    min_kME=0.05,
)

result.U        # (n_cells, k) cell-level program activities
result.V        # (n_genes, k) signed, sparse gene loadings
result.kME      # (n_genes, k) gene-module membership correlations
result.labels   # (n_genes,) assigned module index, or -1 ("grey")
```

For datasets with **n ≥ 100,000 cells**, supply `k` explicitly. Automatic rank selection via parallel analysis is disabled at that scale for computational reasons.

```python
result = fit_slice(
    X,
    k=20,
    sparsity=0.02,
)
```

## Choosing hyperparameters

See `benchmarks/` for the sensitivity sweeps behind these recommendations. These were run with 10 seeds on synthetic data with known ground truth.

### `sparsity`

`sparsity` controls the L1 budget

```text
c = max(sqrt(sparsity * p), 1)
```

Recovery is best when `sparsity` is close to the fraction of genes expected in a program. For example, if a program contains roughly 50 genes out of 2,000, a reasonable starting range is `0.02–0.05`.

Values that are too small can under-capture true hub genes. Values much larger than necessary eventually stop providing additional sparsity control because the L1 budget saturates.

See `benchmarks/sensitivity_sparsity_summary.csv`.

### `min_kME`

`min_kME` controls module assignment.

In the synthetic benchmark, values from `0.1` to `0.5` were stable, with 100% recall and at least 99.8% precision on true module genes. A value of `0.05` was too permissive and assigned more background genes.

See `benchmarks/sensitivity_minkme_summary.csv`.

### `k`

When choosing the number of components, overestimating `k` is safer than underestimating it in our benchmarks.

Extra components beyond the true number caused little measurable harm, while underestimating `k` reduced subspace recovery.

See `benchmarks/sensitivity_k_summary.csv`.

## Biological annotation and WGCNA-style reporting

SLICE also provides utilities for annotating modules, relating programs to traits, and exporting hub-gene tables.

```python
from slice_lca.annotation import (
    load_reference_db,
    annotate_modules,
    export_hub_genes,
)
from slice_lca.bio_outputs import (
    module_trait_table,
    export_hub_network,
)

db = load_reference_db(
    "CellMarker_2.0.xlsx",
    species="Human",
)

annotation = annotate_modules(
    result,
    db,
)

trait_table = module_trait_table(
    result.U,
    clinical_trait,
)

hubs = export_hub_genes(
    result,
    annotation,
)
```

These functions use multiple-testing correction for module annotation and module-trait analysis rather than treating a single best-of-many p-value as an ordinary fixed-threshold test.

Install the additional dependencies with:

```bash
pip install -e ".[bio]"
```

See `CHANGELOG.md` for details on the statistical and implementation fixes made during development.

## Comparison with cNMF

`benchmarks/compare_slice_cnmf.py` runs the actual `cnmf` package:

```text
prepare → factorize → combine → consensus
```

It compares cNMF and SLICE on the same data in the same session rather than using a hand-written approximation of cNMF.

Run the synthetic self-test with:

```bash
python benchmarks/compare_slice_cnmf.py
```

Install the additional dependencies with:

```bash
pip install -e ".[discovery]"
```

### What the current benchmark shows

On the synthetic benchmark with overlapping gene sets but no signed or antagonistic structure, cNMF achieved higher gene-loading recovery (0.999) than SLICE (0.78–0.98 depending on sparsity), while SLICE was approximately 30× faster.

The intended distinction is not that SLICE is better at every task.

SLICE is designed to represent **signed and antagonistic programs**, which non-negative factorization cannot represent directly, while also targeting sparse computation on large single-cell matrices.

These benchmark numbers should not be generalized to other datasets without rerunning the comparison.

## Discovery beyond marker recapitulation

`benchmarks/discovery_toolkit.py` contains three ways to test whether a discovered module contains structure beyond a known marker set.

### Cross-tissue reproducibility

```python
cross_tissue_module_reproducibility()
```

Tests whether a module recurs across independent tissues with a consistent hub-gene signature.

### Held-out marker recovery

```python
held_out_marker_recovery()
```

Tests whether curated marker genes are recovered together in an unsupervised module and identifies additional genes found alongside them.

### GO enrichment beyond known markers

```python
go_enrichment_beyond_markers()
```

Runs pathway enrichment on module hub genes after excluding known marker genes.

This uses the real `gseapy` API but has not been network-tested in the development environment, so test it on an example module before relying on it for a larger analysis.

## Preprocessing and centering

Read this section before fitting real data.

```python
from slice_lca.preprocessing import preprocess
from slice_lca.core import fit_slice

X_ready, gene_names_ready = preprocess(
    X_raw,
    gene_names,
)  # QC -> normalize + log1p -> HVG -> scale

result = fit_slice(
    X_ready,
    gene_names_ready,
    k=15,
    sparsity=0.02,
)
```

`fit_slice` centers the input implicitly by default (`center=True`) and raises `NotPreprocessedError` when the input appears to contain raw counts.

These checks were added after a problem was found in the original implementation: the paper described implicit centering, but the original pipeline did not actually implement it.

This matters particularly for continuous signed or antagonistic axes, which are one of the main reasons to use SLICE instead of a non-negative factorization.

Existing results generated before this fix should be rerun.

### Protecting known genes during HVG selection

Variance-based feature selection can remove real biological signal, particularly for low-expression markers. Genes that you already know are important can be explicitly retained:

```python
X_ready, gene_names_ready = preprocess(
    X_raw,
    gene_names,
    n_top_hvg=3000,
    force_include=["Dclk1", "Trpm5"],
)
```

See `CHANGELOG.md` for the case that motivated `force_include`.

## Robustness options

The following options preserve the old default behavior unless explicitly enabled.

### Multiple restarts

Multiple initializations can reduce the chance of ending up in a poor local optimum:

```python
result = fit_slice(
    X,
    k=15,
    sparsity=0.02,
    n_restarts=4,
)
```

### Convergence diagnostics

You can inspect whether individual components converged:

```python
U, D, V, diagnostics = sparse_svd(
    X,
    k=15,
    sparsity=0.02,
    return_diagnostics=True,
)

print([d["converged"] for d in diagnostics])
```

See `tests/test_multistart.py` and `CHANGELOG.md` for the corresponding regression tests and validation.

## Development

Install the development dependencies:

```bash
pip install -e ".[dev,bio]"
```

Run the test suite:

```bash
pytest
```

The regression tests in `tests/test_recovery.py` pin down the published benchmark numbers. If the algorithm changes, rerun the relevant scripts in `benchmarks/` and update the paper and tests together.

For tutorials, API documentation, and the benchmark walkthroughs, see the [SLICE documentation](https://sizerta.github.io/SLICE/).

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT. See [`LICENSE`](LICENSE).
