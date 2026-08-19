from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(
    "paper_tables/Table1_performance_ranking.csv"
)


df = df.sort_values(
    "Mean_RMSE"
)


plt.figure(
    figsize=(9,5)
)

plt.bar(
    df["Variant"],
    df["Mean_RMSE"]
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.ylabel(
    "Mean RMSE"
)

plt.title(
    "Ablation Performance Ranking"
)

plt.tight_layout()

plt.savefig(
    "paper_tables/rmse_ranking.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Saved rmse_ranking.png")