"""Shared style for all SLICE paper figures -- consistent fonts, colors,
sizing across the whole set. Import this before plotting anything."""
import matplotlib.pyplot as plt
import matplotlib as mpl

COLORS = {
    "SLICE": "#2E5C8A",       # deep blue -- the method
    "SLICE_light": "#7FA8D0",
    "cNMF": "#C9622C",        # burnt orange -- primary competitor
    "NMF": "#C9622C",
    "PCA": "#8B8B8B",         # grey -- classical baseline
    "SparsePCA": "#5B8C5A",   # green
    "center_true": "#2E5C8A",
    "center_false": "#B0392E",
    "grid": "#E0E0E0",
    "text": "#2B2B2B",
}

def set_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
    })
