import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


RESULT_DIR = Path(
    "outputs/v2/ablation_full/results"
)

METRIC = "rmse"


def load_results():

    data = defaultdict(dict)

    for file in RESULT_DIR.glob("*.json"):

        obj = json.loads(
            file.read_text(
                encoding="utf-8"
            )
        )

        task_id = obj["task_id"]
        variant = obj["variant"]

        value = obj["metrics"][METRIC]

        data[task_id][variant] = value

    return pd.DataFrame(data).T


def main():

    df = load_results()

    print("Shape:", df.shape)
    print("\nVariants:")
    print(df.columns.tolist())


    variants = df.columns.tolist()

    arrays = [
        df[v].values
        for v in variants
    ]


    # Friedman test

    stat, p = friedmanchisquare(
        *arrays
    )

    print("\nFriedman Test")
    print("Statistic:", stat)
    print("p-value:", p)


    # Wilcoxon vs PC_NFPSO

    baseline = "PC_NFPSO"

    comparisons = []

    for v in variants:

        if v == baseline:
            continue

        stat, p = wilcoxon(
            df[baseline],
            df[v],
            alternative="two-sided"
        )

        comparisons.append(
            {
                "Comparison":
                    f"{baseline} vs {v}",
                "Statistic": stat,
                "p_value": p,
            }
        )


    result = pd.DataFrame(
        comparisons
    )

    result["p_adjusted"] = multipletests(
        result["p_value"],
        method="holm"
    )[1]


    result.to_csv(
        "wilcoxon_holm_results.csv",
        index=False,
    )


    print("\nSaved:")
    print("wilcoxon_holm_results.csv")

    print(result)


if __name__ == "__main__":
    main()