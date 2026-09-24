import sys
sys.path.insert(0, '.')
from _style import set_style, COLORS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

set_style()

fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.9))

# --- Panel A: sparsity ---
df_s = pd.read_csv("../benchmarks/sensitivity_sparsity_summary.csv")
ax = axes[0]
ax.errorbar(df_s["sparsity"], df_s["W_Corr_mean"], yerr=df_s["W_Corr_sd"],
            marker="o", ms=4, color=COLORS["SLICE"], capsize=3, linewidth=1.4)
ax.set_xscale("log")
ax.set_xlabel("sparsity ($s$)")
ax.set_ylabel("Gene-loading\ncorrelation")
ax.set_title("A. Sparsity budget", fontsize=10, loc="left")
ax.axvspan(0.02, 0.05, color=COLORS["SLICE_light"], alpha=0.25, zorder=0)

# --- Panel B: k (rank misspecification) ---
df_k = pd.read_csv("../benchmarks/sensitivity_k_summary.csv")
ax = axes[1]
ax.errorbar(df_k["k_fit"], df_k["SubspaceErr_mean"], yerr=df_k["SubspaceErr_sd"],
            marker="o", ms=4, color=COLORS["SLICE"], capsize=3, linewidth=1.4)
ax.axvline(5, color=COLORS["NMF"], linestyle="--", linewidth=1.2, label="true $k$")
ax.set_xlabel("fitted $k$")
ax.set_ylabel("Subspace error")
ax.set_title("B. Rank misspecification", fontsize=10, loc="left")
ax.legend(frameon=False, loc="upper right", fontsize=8)

# --- Panel C: min_kME ---
df_m = pd.read_csv("../benchmarks/sensitivity_minkme_summary.csv")
ax = axes[2]
ax.errorbar(df_m["min_kME"], df_m["precision_mean"], yerr=df_m["precision_sd"],
            marker="o", ms=4, color=COLORS["SLICE"], capsize=3, linewidth=1.4, label="precision")
ax.errorbar(df_m["min_kME"], df_m["recall_mean"], yerr=df_m["recall_sd"],
            marker="s", ms=4, color=COLORS["cNMF"], capsize=3, linewidth=1.4, label="recall")
ax.set_xlabel(r"assignment threshold ($\theta$)")
ax.set_ylabel("Module assignment\nprecision / recall")
ax.set_title("C. Assignment threshold", fontsize=10, loc="left")
ax.legend(frameon=False, loc="lower right", fontsize=8)
ax.set_ylim(0.4, 1.05)

fig.tight_layout()
fig.savefig("fig_hyperparam.pdf")
fig.savefig("fig_hyperparam.png")
print("saved fig_hyperparam.pdf/.png")
