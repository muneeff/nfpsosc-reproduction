from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

csv_path = Path("paper_tables/Table1_performance_ranking.csv")
if not csv_path.exists():
    csv_path = Path("outputs/q1_minimal/statistical_tests") / "Table1_performance_ranking.csv"

df = pd.read_csv(csv_path)

# البحث عن عمود MASE أو البديل الأنسب المقاوم لتغير المقاييس
metric_col = next((col for col in df.columns if "mase" in col.lower()), None)
if metric_col is None:
    metric_col = next((col for col in df.columns if "rmse" in col.lower()), df.columns[1])

variant_col = next((col for col in df.columns if "variant" in col.lower() or "model" in col.lower()), df.columns[0])

df = df.sort_values(by=metric_col)

plt.figure(figsize=(9, 5))
plt.bar(df[variant_col], df[metric_col])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Mean MASE (Scale-Independent)")
plt.title("Model Performance Ranking (MASE)")
plt.tight_layout()

output_path = Path("paper_tables/rmse_ranking.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print("Updated rmse_ranking.png successfully using scale-independent MASE!")