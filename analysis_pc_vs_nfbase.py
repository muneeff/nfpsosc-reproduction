import json
from pathlib import Path
import numpy as np


ROOT = Path(
    "outputs/v2/ablation_full/results"
)


data = {}

for f in ROOT.glob("*.json"):

    x = json.loads(
        f.read_text(encoding="utf-8")
    )

    if x["variant"] in ["PC_NFPSO", "NF_BASE"]:

        data.setdefault(
            x["task_id"],
            {}
        )[x["variant"]] = x["metrics"]["rmse"]


wins = []
losses = []

for task, v in data.items():

    if "PC_NFPSO" not in v:
        continue

    if "NF_BASE" not in v:
        continue

    diff = v["NF_BASE"] - v["PC_NFPSO"]

    if diff > 0:
        wins.append(diff)
    else:
        losses.append(abs(diff))


print("PC wins:", len(wins))
print("PC losses:", len(losses))

print()

print(
    "Average improvement when PC wins:",
    np.mean(wins)
)

print(
    "Average degradation when PC loses:",
    np.mean(losses)
)

print()

print(
    "Total mean difference NF_BASE-PC:",
    np.mean(
        [
            v["NF_BASE"] - v["PC_NFPSO"]
            for v in data.values()
        ]
    )
)