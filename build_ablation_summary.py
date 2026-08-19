import json
from pathlib import Path
from collections import defaultdict
import pandas as pd


RESULT_DIR = Path(
    "outputs/v2/ablation_full/results"
)

OUTPUT = Path(
    "ablation_summary.csv"
)


def main():

    data = defaultdict(list)

    for file in RESULT_DIR.glob("*.json"):

        obj = json.loads(
            file.read_text(
                encoding="utf-8"
            )
        )

        variant = obj["variant"]
        metrics = obj["metrics"]

        data[variant].append(metrics)


    rows = []

    for variant, items in data.items():

        df = pd.DataFrame(items)

        row = {
            "Variant": variant,
            "Count": len(df),
        }

        for col in df.columns:
            row[f"Mean_{col}"] = df[col].mean()
            row[f"Std_{col}"] = df[col].std()

        rows.append(row)


    summary = pd.DataFrame(rows)
    print("Columns:", summary.columns.tolist())
    if "Mean_RMSE" in summary.columns:
       	    summary = summary.sort_values(
       	        	by="Mean_RMSE"
            )
    summary.to_csv(
        OUTPUT,
        index=False,
    )

    print(summary)
    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()