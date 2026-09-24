# SLICE

SLICE decomposes a (cells × genes) expression matrix into sparse, signed
gene programs and their per-cell activity scores. It never densifies
sparse input and never constructs a gene × gene correlation matrix, so
it scales to atlas-sized datasets (the largest benchmark in the paper
is 897,733 cells) with a memory footprint that stays close to the size
of the input itself.

"Signed" is the part that matters mechanically. Non-negative methods
(NMF, cNMF) cannot represent a component where one gene set goes up
while another goes down — the constraint `W, H ≥ 0` makes it
mathematically impossible, not just difficult. On real intestinal
epithelium data, this isn't a hypothetical: a single SLICE component
resolves a coherent oxidative-phosphorylation program (`Cox4i1`,
`Atp5b`, `Slc25a5`, ...) opposed to Paneth-cell secretory identity
(`Lyz1`, `Defa26`, `Clps`, ...) as one signed axis. Running the actual
`cnmf` package on the same data, the same genes split across disjoint,
non-overlapping components — not because cNMF is a worse
implementation, but because it structurally cannot do otherwise. See
[Comparisons](comparisons.md) for the full result, including where
SLICE does *not* have an advantage.

## Where to start

- **New to SLICE?** → [Quickstart](quickstart.md)
- **About to run this on real data?** → read [Preprocessing](preprocessing.md)
  first. Skipping this step is the single most common way to get
  degenerate results, and `fit_slice` will refuse to run on raw counts
  rather than fail silently — but it's worth understanding why.
- **Trying to reproduce a specific number from the paper?** →
  [Reproducing the paper](reproducing-paper.md)
- **Deciding whether SLICE is the right tool for your data?** →
  [Comparisons against other methods](comparisons.md)

## What this is not

SLICE assumes an approximately linear latent structure. It does not
model non-linear manifolds (that's what scVI and similar deep
generative approaches are for), and it does not causally validate
anything it finds — a signed axis is a statistical association, not
evidence of regulatory repression between the genes on opposite poles.
The [paper's](PLACEHOLDER) Limitations section (§4.3) goes through
this, and the caveats that matter for actually using the tool are
repeated where they're relevant in these docs rather than left in one
section to be forgotten.
