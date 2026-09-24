import sys
sys.path.insert(0, '.')
from _style import set_style, COLORS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

set_style()
df = pd.read_csv("../benchmarks/centering_fix_raw.csv")

fig, ax = plt.subplots(figsize=(4.0, 3.2))

conditions = [False, True]
labels = ["Not centered\n(matches original\nreal-data pipeline)", "Implicitly centered\n(center=True)"]
means = [df[df.center == c]["gene_corr"].mean() for c in conditions]
sds = [df[df.center == c]["gene_corr"].std() for c in conditions]
colors = [COLORS["center_false"], COLORS["center_true"]]

x = np.arange(2)
bars = ax.bar(x, means, yerr=sds, capsize=4, color=colors, width=0.55,
               edgecolor="white", linewidth=0.5, zorder=3,
               error_kw={"linewidth": 1.2, "ecolor": "#333333"})

# overlay individual seeds as jittered points
rng = np.random.default_rng(0)
for i, c in enumerate(conditions):
    vals = df[df.center == c]["gene_corr"].values
    jitter = rng.uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(np.full(len(vals), x[i]) + jitter, vals, s=14, color="white",
               edgecolor="#333333", linewidth=0.6, zorder=4, alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Gene-loading correlation\nwith true signed axis")
ax.set_ylim(0, 1.05)
ax.set_title("Recovery of a known antagonistic axis\n(8 seeds, identical input matrix)", fontsize=10)

for i, (m, s) in enumerate(zip(means, sds)):
    ax.text(x[i], m + s + 0.05, f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")

fig.tight_layout()
fig.savefig("fig_centering.pdf")
fig.savefig("fig_centering.png")
print("saved fig_centering.pdf/.png")
