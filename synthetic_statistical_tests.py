from pathlib import Path
import pandas as pd
import numpy as np

from scipy.stats import friedmanchisquare, wilcoxon


FILE = Path(
    "outputs/v2/final_external_memory_safe/analysis/final_results_master.csv"
)

df = pd.read_csv(FILE)

df = df[df["domain"] == "synthetic"].copy()


models = sorted(df["method"].unique())

# One row = one synthetic condition
keys = [
    "generator",
    "noise_level",
    "dgp_seed"
]


pivot = (
    df.pivot_table(
        index=keys,
        columns="method",
        values="rmse",
        aggfunc="mean"
    )
    .dropna()
)


print("Synthetic cases =", len(pivot))
print("Models =", list(pivot.columns))


# Friedman

stat, p = friedmanchisquare(
    *[
        pivot[m].values
        for m in pivot.columns
    ]
)

print("\nFRIEDMAN")
print("stat =", stat)
print("p =", p)


# Wilcoxon PC-NFPSO comparisons

pc = pivot["pc_nfpso"]

print("\nWILCOXON")

for m in models:

    if m == "pc_nfpso":
        continue

    stat, p = wilcoxon(
        pc,
        pivot[m],
        alternative="less"
    )

    print(
        f"pc_nfpso vs {m}: "
        f"stat={stat:.3f}, p={p:.6f}"
    )