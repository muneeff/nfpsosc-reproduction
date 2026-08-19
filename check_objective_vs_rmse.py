import json
from pathlib import Path
import numpy as np


ROOT = Path("outputs/v2/ablation_full/results")


objs = []
rmses = []

for f in ROOT.glob("*.json"):

    x = json.loads(
        f.read_text(encoding="utf-8")
    )

    if x.get("variant") == "PC_NFPSO":

        if x.get("best_objective") is not None:

            objs.append(
                x["best_objective"]
            )

            rmses.append(
                x["metrics"]["rmse"]
            )


print("Samples:", len(objs))

if len(objs) > 1:
    corr = np.corrcoef(
        objs,
        rmses
    )[0,1]

    print("Correlation best_objective vs RMSE:", corr)

print("Objective mean:", np.mean(objs))
print("RMSE mean:", np.mean(rmses))