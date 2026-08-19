import json
from pathlib import Path
from collections import defaultdict
import numpy as np


ROOT = Path(
    "outputs/v2/ablation_full/results"
)


def main():

    data = defaultdict(lambda: defaultdict(list))

    for f in ROOT.glob("*.json"):

        obj = json.loads(
            f.read_text(
                encoding="utf-8"
            )
        )

        data[obj["variant"]]["rmse"].append(
            obj["metrics"]["rmse"]
        )


    print(
        f"{'Variant':25s}"
        f"{'Mean':12s}"
        f"{'Median':12s}"
        f"{'P90':12s}"
        f"{'Std':12s}"
    )

    print("-"*75)

    for variant, values in data.items():

        rmse = np.array(
            values["rmse"],
            dtype=float
        )

        print(
            f"{variant:25s}"
            f"{rmse.mean():12.3f}"
            f"{np.median(rmse):12.3f}"
            f"{np.percentile(rmse,90):12.3f}"
            f"{rmse.std():12.3f}"
        )


if __name__ == "__main__":
    main()