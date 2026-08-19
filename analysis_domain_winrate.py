import json
from pathlib import Path
from nfpsosc.v2.final_runner import load_final_manifest


ROOT = Path("outputs/v2/ablation_full/results")


from pathlib import Path
import json

manifest_path = next(
    Path(".").rglob("final_external_manifest.json")
)

print("Using manifest:", manifest_path)

with open(manifest_path, encoding="utf-8") as f:
    manifest_data = json.load(f)

domain_map = {
    row["run_id"]: row["domain"]
    for row in manifest_data["tasks"]
}

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


result = {
    "synthetic": {"pc":0, "nf":0},
    "real": {"pc":0, "nf":0},
}


for task, values in scores.items():

    if "PC_NFPSO" not in values:
        continue

    if "NF_BASE" not in values:
        continue

    domain = domain_map.get(task)

    if domain is None:
        continue

    if values["PC_NFPSO"] < values["NF_BASE"]:
        result[domain]["pc"] += 1
    else:
        result[domain]["nf"] += 1


print(result)