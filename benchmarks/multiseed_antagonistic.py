import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings("ignore")
from slice_lca.datasets import make_antagonistic_axis
from slice_lca._eval import evaluate_antagonistic_efficiency

N_SEEDS = 10
rows = []
t0 = time.time()
for sd in range(N_SEEDS):
    X, S, W = make_antagonistic_axis(n_cells=2000, n_genes=1000, snr=5.0, seed=sd)
    df = evaluate_antagonistic_efficiency(X, S, W, nmf_seed=sd)
    df["Seed"] = sd
    rows.append(df)
all_df = pd.concat(rows, ignore_index=True)
all_df.to_csv("multiseed_antagonistic_raw.csv", index=False)

summary = all_df.groupby(["Method", "k"]).agg(
    MSE_mean=("MSE", "mean"), MSE_sd=("MSE", "std"),
    StateCorr_mean=("State_Corr", "mean"), StateCorr_sd=("State_Corr", "std"),
).reset_index()
print(f"{N_SEEDS}-seed summary (mean +/- SD):")
print(summary.round(6).to_string(index=False))
print(f"\ntotal time: {time.time()-t0:.0f}s")
