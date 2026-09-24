# Biological annotation and WGCNA-style reporting

Two things WGCNA users typically expect that a bare `fit_slice()` call
doesn't give you: a cell-type or biological label per module, and a
module-trait correlation grid. Both exist here as separate,
composable functions rather than being baked into `fit_slice` itself.

```bash
pip install -e ".[bio]"
```

## Naming modules by cell type

```python
from slice_lca.annotation import load_reference_db, annotate_modules, export_hub_genes

db = load_reference_db("CellMarker_2.0.xlsx", species="Human")
annotation = annotate_modules(result, db)
hubs = export_hub_genes(result, annotation)
```

`load_reference_db` auto-detects which columns in your reference file
hold cell types, gene symbols, and species — column naming varies
enough across different CellMarker-style exports that this is
inherently a little fragile. It prints exactly which column it picked
for each role; read that output the first time you point it at a new
file, especially if the annotations that come back look wrong.

`annotate_modules` tests each module's top hub genes against every
cell type in the reference via hypergeometric enrichment, and
Benjamini-Hochberg-corrects across the whole family of tests before
calling anything identified. This matters more than it might look:
against a realistic ~120-cell-type reference with zero real signal,
taking the single best p-value per module against a fixed uncorrected
threshold gives roughly a 5% chance *per module* of confidently naming
something that isn't there
(`tests/test_annotation.py::test_no_false_positive_identification_on_noise`
reproduces this directly on synthetic noise). A module only gets
labeled here if its best BH-adjusted q-value clears `q_thresh`
(default 0.05) — unlabeled modules are a correct, honest outcome when
the enrichment isn't strong enough to say more.

## Module-trait correlation

The WGCNA-familiar workflow: correlate each module's cell-level
activity against an external trait — clinical, experimental, whatever
you have per-cell or per-sample values for.

```python
from slice_lca.bio_outputs import module_trait_table, plot_module_trait_heatmap

trait_table = module_trait_table(result.U, clinical_trait)
plot_module_trait_heatmap(trait_table, trait_name="survival_time")
```

`module_trait_table` runs a BH-corrected Pearson correlation between
every module and the trait vector you pass. For a categorical trait
with more than two levels (a multi-class subtype label, say), the
standard approach is to convert it into a set of binary indicator
variables first and call this once per indicator — the function
itself expects one continuous or binary vector at a time rather than
guessing at how to handle an arbitrary categorical column.

## Other reporting utilities

```python
from slice_lca.bio_outputs import hub_gene_table, export_hub_network, gene_module_profile, variance_explained

hub_gene_table(result.kME, result.gene_names, result.labels, module_id=1, top_n=20)
export_hub_network(result, module_id=1, path="module1_network.tsv")   # for Cytoscape/Gephi import
gene_module_profile(result.kME, result.gene_names, gene="Lyz1")        # which modules does one gene load on?
variance_explained(result.D, X_ready)
```

`gene_module_profile` is worth knowing about specifically if you're
checking whether a single gene of interest shows up where you'd
expect — pass a gene symbol and get back its `kME` across every
module in the fit, sorted by strength, rather than searching the full
matrix yourself.
