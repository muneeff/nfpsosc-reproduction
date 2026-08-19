import json
from pathlib import Path
import numpy as np


ROOT = Path("outputs/v2/ablation_full/results")


scores = {}

for f in ROOT.glob("*.json"):

    x = json.loads(
        f.read_text(encoding="utf-8")
    )

    if x["variant"] in ["PC_NFPSO", "NF_BASE"]:

        scores.setdefault(
            x["task_id"],
            {}
        )[x["variant"]] = x["metrics"]["rmse"]


rows = []

for task, v in scores.items():

    if "PC_NFPSO" in v and "NF_BASE" in v:

        rows.append(
            (
                task,
                v["NF_BASE"],
                v["PC_NFPSO"],
                v["NF_BASE"] - v["PC_NFPSO"]
            )
        )


rows.sort(
    key=lambda x: x[1]
)


n = len(rows)
quartile_size = n // 4


for i, name in enumerate(
    ["Q1 Easy", "Q2", "Q3", "Q4 Hard"]
):

    start = i * quartile_size

    if i == 3:
        group = rows[start:]
    else:
        group = rows[start:start+quartile_size]


    wins = sum(
        1
        for r in group
        if r[3] > 0
    )

    mean_diff = np.mean(
        [r[3] for r in group]
    )

    print(name)
    print("Count:", len(group))
    print("PC wins:", wins)
    print(
        "Win rate:",
        wins / len(group) * 100
    )
    print(
        "Mean NF_BASE-PC RMSE:",
        mean_diff
    )
    print("-"*40)