import pandas as pd

path="outputs/v2/final_external_memory_safe/analysis/final_results_master.csv"

df=pd.read_csv(path)

s=df[df.domain=="synthetic"].copy()

# aggregate optimizer replicates
agg_keys=[
    "generator",
    "dgp_seed",
    "noise_level",
    "method"
]

agg=(
    s.groupby(agg_keys)["mase"]
    .median()
    .reset_index()
)

pc=(
    agg[agg.method=="pc_nfpso"]
    .rename(columns={"mase":"pc_mase"})
    .drop(columns="method")
)

rows=[]

for model in sorted(agg.method.unique()):
    if model=="pc_nfpso":
        continue

    base=(
        agg[agg.method==model]
        .rename(columns={"mase":"base_mase"})
        .drop(columns="method")
    )

    m=pc.merge(
        base,
        on=["generator","dgp_seed","noise_level"]
    )

    rows.append([
        model,
        (m.pc_mase<m.base_mase).sum(),
        (m.pc_mase>m.base_mase).sum(),
        len(m)
    ])

print(pd.DataFrame(
    rows,
    columns=["model","PC_win","PC_loss","N"]
))