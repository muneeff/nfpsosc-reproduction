import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


path = "outputs/v2/final_external_memory_safe/analysis/final_results_master.csv"

df = pd.read_csv(path)

# Synthetic only
df = df[df["domain"] == "synthetic"].copy()


# Aggregate optimizer replicates
keys = [
    "generator",
    "dgp_seed",
    "noise_level",
    "method"
]

agg = (
    df.groupby(keys)["mase"]
    .median()
    .reset_index()
)


# Pivot
pivot = (
    agg
    .pivot_table(
        index=[
            "generator",
            "dgp_seed",
            "noise_level"
        ],
        columns="method",
        values="mase"
    )
    .dropna()
)


print("CASES =", len(pivot))


# =========================
# Average Rank
# =========================

print("\nAVERAGE RANK")

ranks = pivot.rank(
    axis=1,
    method="average"
)

print(
    ranks.mean()
    .sort_values()
    .round(3)
)


# =========================
# Friedman
# =========================

print("\nFRIEDMAN")

stat, p = friedmanchisquare(
    *[
        pivot[c].values
        for c in pivot.columns
    ]
)

print("stat =", stat)
print("p =", p)



# =========================
# Wilcoxon
# =========================

print("\nWILCOXON PC-NFPSO")

pc = pivot["pc_nfpso"]

rows=[]

for model in pivot.columns:

    if model=="pc_nfpso":
        continue

    diff = pc - pivot[model]

    w,pv = wilcoxon(
        diff,
        alternative="two-sided"
    )

    rows.append(
        [
            model,
            w,
            pv,
            diff.median()
        ]
    )


res=pd.DataFrame(
    rows,
    columns=[
        "model",
        "stat",
        "raw_p",
        "median_difference"
    ]
)


# Holm correction

res=res.sort_values("raw_p")

m=len(res)

res["holm_p"]=[
    min((m-i)*p,1.0)
    for i,p in enumerate(res.raw_p)
]


print(
    res.to_string(index=False)
)
