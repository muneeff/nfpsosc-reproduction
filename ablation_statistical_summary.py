import pandas as pd

summary = pd.read_csv(
    "ablation_summary.csv"
)

stats = pd.read_csv(
    "wilcoxon_holm_results.csv"
)

# ترتيب حسب RMSE
summary = summary.sort_values(
    "Mean_rmse"
)

summary.to_csv(
    "ablation_final_table.csv",
    index=False
)

stats.to_csv(
    "ablation_significance_table.csv",
    index=False
)

print("Saved:")
print("ablation_final_table.csv")
print("ablation_significance_table.csv")

print("\nRanking:")
print(
    summary[
        [
            "Variant",
            "Mean_rmse",
            "Mean_mae",
            "Mean_mase"
        ]
    ]
)