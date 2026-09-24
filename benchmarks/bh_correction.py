"""
Benjamini-Hochberg correction for the 15-module TCGA log-rank scan
(Real_DealV4 notebook, cell 41). Uses the exact p-values already
printed in that cell's output -- no new data needed.
"""
import pandas as pd
from statsmodels.stats.multitest import multipletests

data = [
    ("M10", 0.0000), ("M13", 0.0001), ("M11", 0.0007), ("M2", 0.0018),
    ("M1", 0.0021), ("M6", 0.0023), ("M5", 0.0055), ("M0", 0.0055),
    ("M3", 0.0056), ("M4", 0.0175), ("M7", 0.0225), ("M12", 0.0305),
    ("M8", 0.0409), ("M9", 0.3000), ("M14", 0.3018),
]
df = pd.DataFrame(data, columns=["Module", "p_raw"])
# M10 was reported as 0.0000 (rounded); use a conservative stand-in for the
# correction arithmetic only -- doesn't change any conclusion.
df.loc[df.p_raw == 0.0, "p_raw"] = 0.00005

reject, q, _, _ = multipletests(df.p_raw, alpha=0.05, method="fdr_bh")
df["q_BH"] = q
df["significant_FDR05"] = reject
df = df.sort_values("p_raw")
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(df.to_string(index=False))
print(f"\n{reject.sum()}/15 modules remain significant at FDR<0.05")
print(f"M1 (the flagship figure): q = {df.loc[df.Module=='M1','q_BH'].values[0]:.4f}")
