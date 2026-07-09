from pathlib import Path
import pandas as pd

root = Path(r"outputs\q1_minimal\statistical_tests_final_r1p00\synthetic_unified_benchmarks")
out = Path(r"outputs\q1_minimal\paper_tables")
out.mkdir(parents=True, exist_ok=True)

metrics = ["rmse", "mae", "smape_percent", "mase"]

summary_rows = []
comparison_rows = []

for metric in metrics:
    summary_path = root / metric / f"{metric}_model_summary.csv"
    comparison_path = root / metric / f"{metric}_reference_comparisons.csv"

    summary = pd.read_csv(summary_path)
    summary["metric"] = metric
    summary_rows.append(summary)

    comparison = pd.read_csv(comparison_path)
    comparison["metric"] = metric
    comparison_rows.append(comparison)

summary_all = pd.concat(summary_rows, ignore_index=True)
comparisons_all = pd.concat(comparison_rows, ignore_index=True)

summary_out = out / "synthetic_metric_summary_all.csv"
comparisons_out = out / "synthetic_pairwise_holm_all.csv"

summary_all.to_csv(summary_out, index=False)
comparisons_all.to_csv(comparisons_out, index=False)

summary_cols = [
    "metric", "model", "mean", "std", "median", "rank_mean",
    "rank_median", "n_conditions"
]
summary_all[summary_cols].to_csv(out / "paper_table_synthetic_metric_summary.csv", index=False)

comparison_cols = [
    "metric", "challenger_model", "direction_by_mean", "mean_difference",
    "ci_low", "ci_high", "p_adjusted_holm", "reject_alpha_holm"
]
comparisons_all[comparison_cols].to_csv(out / "paper_table_synthetic_pairwise_holm.csv", index=False)

print("[DONE] created paper tables:")
print(summary_out)
print(comparisons_out)
print(out / "paper_table_synthetic_metric_summary.csv")
print(out / "paper_table_synthetic_pairwise_holm.csv")
