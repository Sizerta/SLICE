import sys
sys.path.insert(0, '.')
from _style import set_style, COLORS
import matplotlib.pyplot as plt
import numpy as np

set_style()
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))

# --- Panel A: runtime, real datasets (log scale) ---
datasets = ["Hard-overlap\n(synthetic)", "Haber\nintestine", "Pancreas"]
slice_t = [0.29, 26.4, 16.3]
cnmf_t = [11.92, 209.9, 205.3]

ax = axes[0]
x = np.arange(len(datasets))
w = 0.35
ax.bar(x - w/2, slice_t, width=w, label="SLICE", color=COLORS["SLICE"], zorder=3)
ax.bar(x + w/2, cnmf_t, width=w, label="cNMF", color=COLORS["cNMF"], zorder=3)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=8.5)
ax.set_ylabel("Runtime (s, log scale)")
ax.set_title("A. Runtime, same hardware", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8.5)
for i, (s, c) in enumerate(zip(slice_t, cnmf_t)):
    ax.text(i, max(s, c) * 1.3, f"{c/s:.0f}x", ha="center", fontsize=8, fontweight="bold", color="#333")

# --- Panel B: cross-method agreement distributions on real data ---
haber = [0.676, 0.549, 0.530, 0.482, 0.420, 0.379, 0.335, 0.296, 0.259, 0.255, 0.248, 0.217, 0.213, 0.194, 0.148]
pancreas = [0.494, 0.492, 0.486, 0.438, 0.407, 0.403, 0.319, 0.273, 0.246, 0.236, 0.197, 0.166, 0.129, 0.108, 0.031]
synthetic_selftest = [0.988, 0.793, 0.187, 0.031]  # hard-overlap self-test, k=4

ax = axes[1]
positions = [1, 2, 3]
data = [synthetic_selftest, haber, pancreas]
labels = ["Synthetic\nself-test", "Haber\nintestine", "Pancreas"]
bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                 medianprops={"color": "#333333", "linewidth": 1.4},
                 boxprops={"facecolor": COLORS["SLICE_light"], "edgecolor": "#333333", "linewidth": 0.8},
                 whiskerprops={"color": "#333333"}, capprops={"color": "#333333"},
                 flierprops={"markersize": 0})
rng = np.random.default_rng(0)
for pos, vals in zip(positions, data):
    jitter = rng.uniform(-0.09, 0.09, size=len(vals))
    ax.scatter(np.full(len(vals), pos) + jitter, vals, s=12, color=COLORS["SLICE"],
               alpha=0.6, zorder=4, edgecolor="none")
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("|Gene-loading correlation|\nper matched component")
ax.set_title("B. Cross-method agreement", fontsize=10, loc="left")
ax.set_ylim(-0.05, 1.05)

fig.tight_layout()
fig.savefig("fig_cnmf_comparison.pdf")
fig.savefig("fig_cnmf_comparison.png")
print("saved fig_cnmf_comparison.pdf/.png")
