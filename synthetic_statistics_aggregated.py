import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

path="outputs/v2/final_external_memory_safe/analysis/final_results_master.csv"

df=pd.read_csv(path)

# Synthetic only
s=df[df.domain=="synthetic"].copy()

# Aggregate optimizer replicates
keys=[
    "generator",
    "dgp_seed",
    "noise_level",
    "method"
]

agg=(
    s.groupby(keys)["mase"]
    .median()
    .reset_index()
)

# Pivot: rows = experimental cases, columns = models
pivot=(
    agg
    .pivot_table(
        index=["generator","dgp_seed","noise_level"],
        columns="method",
        values="mase"
    )
    .dropna()
)

print("CASES =",len(pivot))
print()

# Average rank
ranks=pivot.rank(axis=1,method="average")

print("AVERAGE RANK")
print(
    ranks.mean()
    .sort_values()
    .to_string()
)

print("\nFRIEDMAN")

stat,p=friedmanchisquare(
    *[
        pivot[c].values
        for c in pivot.columns
    ]
)

print("stat =",stat)
print("p =",p)


print("\nPC-NFPSO WILCOXON")

pc=pivot["pc_nfpso"]

results=[]

for model in pivot.columns:

    if model=="pc_nfpso":
        continue

    diff=pc-pivot[model]

    w,pv=wilcoxon(
        diff,
        alternative="two-sided"
    )

    results.append(
        [
            model,
            w,
            pv,
            diff.median()
        ]
    )

res=pd.DataFrame(
    results,
    columns=[
        "comparison",
        "stat",
        "p_raw",
        "median_difference"
    ]
)

print(res.to_string(index=False))