import pandas as pd
import numpy as np

from scipy.stats import wilcoxon


FILE = "outputs/v2/final_external_memory_safe/analysis/final_results_master.csv"


df = pd.read_csv(FILE)

df = df[df["domain"] == "synthetic"].copy()


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


pc = pivot["pc_nfpso"]


results = []


def holm_adjust(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.zeros(len(pvalues))

    for i, idx in enumerate(order):
        adjusted[idx] = min(
            (len(pvalues)-i) * pvalues[idx],
            1.0
        )

    return adjusted


def cliffs_delta(x, y):
    more = 0
    less = 0

    for a in x:
        for b in y:
            if a > b:
                more += 1
            elif a < b:
                less += 1

    return (more - less) / (len(x)*len(y))


raw_p = []
names = []

for m in pivot.columns:

    if m == "pc_nfpso":
        continue

    stat, p = wilcoxon(
        pc,
        pivot[m],
        alternative="less"
    )

    raw_p.append(p)
    names.append(m)


adj = holm_adjust(raw_p)


for m, p, hp in zip(names, raw_p, adj):

    delta = cliffs_delta(
        pc.values,
        pivot[m].values
    )

    results.append({
        "comparison": f"pc_nfpso vs {m}",
        "raw_p": p,
        "holm_p": hp,
        "cliffs_delta": delta
    })


out = pd.DataFrame(results)

print(out.to_string(index=False))

out.to_csv(
    "outputs/v2/final_external_memory_safe/analysis/synthetic_posthoc_statistics.csv",
    index=False
)