import json
from pathlib import Path


ROOT = Path(
    "outputs/v2/ablation_test/results"
)


scores = {}

for f in ROOT.glob("*.json"):

    x = json.loads(
        f.read_text(
            encoding="utf-8"
        )
    )

    if x["variant"] in ["PC_NFPSO", "NF_BASE"]:

        scores.setdefault(
            x["task_id"],
            {}
        )[x["variant"]] = x["metrics"]["rmse"]


pc = 0
nf = 0
ties = 0

for task, values in scores.items():

    if "PC_NFPSO" not in values:
        continue

    if "NF_BASE" not in values:
        continue

    if values["PC_NFPSO"] < values["NF_BASE"]:
        pc += 1

    elif values["NF_BASE"] < values["PC_NFPSO"]:
        nf += 1

    else:
        ties += 1


total = pc + nf + ties

print("Compared:", total)
print("PC_NFPSO wins:", pc)
print("NF_BASE wins:", nf)
print("Ties:", ties)

if total:
    print(
        "PC win rate:",
        pc / total * 100,
        "%"
    )