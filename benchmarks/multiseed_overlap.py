import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings("ignore")
from slice_lca._eval import benchmark_methods_comparison

N_SEEDS = 10
overlaps = [0.2, 0.5, 0.8]
seeds = list(range(N_SEEDS))

t0 = time.time()
rows = []
for ov in overlaps:
    for sd in seeds:
        df = benchmark_methods_comparison(overlap_frac=ov, snr=5.0, seed=sd)
        df["Overlap"] = ov
        df["Seed"] = sd
        rows.append(df)
    print(f"overlap={ov} done ({time.time()-t0:.0f}s elapsed)")

all_df = pd.concat(rows, ignore_index=True)
all_df.to_csv("multiseed_overlap_raw.csv", index=False)

summary = all_df.groupby(["Overlap", "Method"]).agg(
    W_Corr_mean=("Avg_W_Corr", "mean"), W_Corr_sd=("Avg_W_Corr", "std"),
    Neg_Recall_mean=("Avg_Neg_Recall", "mean"), Neg_Recall_sd=("Avg_Neg_Recall", "std"),
    Subspace_Error_mean=("Subspace_Error", "mean"), Subspace_Error_sd=("Subspace_Error", "std"),
).reset_index()
summary.to_csv("multiseed_overlap_summary.csv", index=False)
pd.set_option("display.width", 140)
print(f"\n{N_SEEDS}-seed summary (mean +/- SD):")
print(summary.round(3).to_string(index=False))
print(f"\ntotal time: {time.time()-t0:.0f}s")
