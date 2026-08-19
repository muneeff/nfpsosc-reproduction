from pathlib import Path
import pandas as pd


INPUT = Path(
    "wilcoxon_holm_results.csv"
)

OUTPUT = Path(
    "paper_tables/Table2_statistical_significance.csv"
)


df = pd.read_csv(INPUT)


# ترتيب حسب p-adjusted
df = df.sort_values(
    "p_adjusted"
)


# إضافة قرار الدلالة
df["Significant"] = (
    df["p_adjusted"] < 0.05
)


df.to_csv(
    OUTPUT,
    index=False
)


print(df.to_string(index=False))

print("\nSaved:", OUTPUT)