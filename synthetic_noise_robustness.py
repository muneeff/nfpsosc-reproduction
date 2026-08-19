import pandas as pd


FILE = "outputs/v2/final_external_memory_safe/analysis/final_results_master.csv"


df = pd.read_csv(FILE)

df = df[df["domain"] == "synthetic"].copy()


# RMSE by noise level
rmse_table = (
    df.groupby(
        ["noise_level", "method"]
    )["rmse"]
    .mean()
    .reset_index()
)


print("\n=== RMSE BY NOISE LEVEL ===\n")

print(
    rmse_table
    .pivot(
        index="method",
        columns="noise_level",
        values="rmse"
    )
    .sort_values(0.05)
)


# Ranking

rank_df = df.copy()

rank_df["rank"] = (
    rank_df
    .groupby(
        ["generator","noise_level"]
    )["rmse"]
    .rank()
)


print("\n=== AVERAGE RANK BY NOISE ===\n")

print(
    rank_df
    .groupby(
        ["noise_level","method"]
    )["rank"]
    .mean()
    .unstack()
)


# degradation

pivot = (
    rmse_table
    .pivot(
        index="method",
        columns="noise_level",
        values="rmse"
    )
)


pivot["degradation_0.05_to_0.20"] = (
    (pivot[0.20]-pivot[0.05])
    /
    pivot[0.05]
)


print("\n=== DEGRADATION ===\n")

print(
    pivot["degradation_0.05_to_0.20"]
    .sort_values()
)