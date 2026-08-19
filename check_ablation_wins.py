import json
from pathlib import Path
from collections import defaultdict

root = Path("outputs/v2/ablation_full/results")

wins = defaultdict(int)

for f in root.glob("*.json"):
    x = json.loads(f.read_text(encoding="utf-8"))

    if x["variant"] in ["NF_BASE", "PC_NFPSO"]:
        wins.setdefault(
            x["task_id"],
            {}
        )[x["variant"]] = x["metrics"]["rmse"]


pc = 0
nf = 0

for task, values in wins.items():
    if len(values) == 2:
        if values["PC_NFPSO"] < values["NF_BASE"]:
            pc += 1
        else:
            nf += 1

print("PC_NFPSO wins:", pc)
print("NF_BASE wins:", nf)
print("Total compared:", pc+nf)