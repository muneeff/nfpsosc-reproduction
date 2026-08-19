from pathlib import Path
import json
import pandas as pd
import numpy as np


ROOT = Path(
    "outputs/v2/ablation_full/results"
)

OUT = Path(
    "paper_tables"
)

OUT.mkdir(
    exist_ok=True
)


# =========================
# Load results
# =========================

rows = []

for f in ROOT.glob("*.json"):

    data = json.loads(
        f.read_text(
            encoding="utf-8"
        )
    )

    if data.get("status") != "success":
        continue

    metric = data["metrics"]

    rows.append(
        {
            "Variant": data["variant"],
            "MASE": metric["mase"],
            "MAE": metric["mae"],
            "RMSE": metric["rmse"],
            "sMAPE": metric["smape"],
            "RMSSE": metric["rmsse"],
        }
    )


df = pd.DataFrame(rows)


# =========================
# Table 1
# Performance ranking
# =========================

summary = (
    df.groupby("Variant")
    .agg(
        Count=("RMSE", "count"),
        Mean_RMSE=("RMSE", "mean"),
        Std_RMSE=("RMSE", "std"),
        Mean_MAE=("MAE", "mean"),
        Std_MAE=("MAE", "std"),
        Mean_MASE=("MASE", "mean"),
        Std_MASE=("MASE", "std"),
        Mean_sMAPE=("sMAPE", "mean"),
        Mean_RMSSE=("RMSSE", "mean"),
    )
    .reset_index()
)


summary = summary.sort_values(
    "Mean_RMSE"
)


summary.to_csv(
    OUT / "Table1_performance_ranking.csv",
    index=False,
)


print("\nTABLE 1")
print(summary.to_string(index=False))


# =========================
# Table 3
# PC_NFPSO vs NF_BASE
# =========================

pivot = {}

for _, row in df[
    df["Variant"].isin(
        ["PC_NFPSO", "NF_BASE"]
    )
].iterrows():

    # no task id in summary rows
    pass


raw = {}

for f in ROOT.glob("*.json"):

    data = json.loads(
        f.read_text(
            encoding="utf-8"
        )
    )

    if data.get("variant") in [
        "PC_NFPSO",
        "NF_BASE",
    ]:

        raw.setdefault(
            data["task_id"],
            {}
        )[data["variant"]] = data["metrics"]["rmse"]


pc_win = 0
nf_win = 0
ties = 0


for task, values in raw.items():

    if (
        "PC_NFPSO" not in values
        or "NF_BASE" not in values
    ):
        continue


    if values["PC_NFPSO"] < values["NF_BASE"]:
        pc_win += 1

    elif values["NF_BASE"] < values["PC_NFPSO"]:
        nf_win += 1

    else:
        ties += 1


win_table = pd.DataFrame(
    [
        {
            "Comparison":
                "PC_NFPSO vs NF_BASE",
            "PC_NFPSO_Wins":
                pc_win,
            "NF_BASE_Wins":
                nf_win,
            "Ties":
                ties,
            "Total":
                pc_win + nf_win + ties,
            "PC_Win_Rate_%":
                100 * pc_win /
                (pc_win + nf_win + ties),
        }
    ]
)


win_table.to_csv(
    OUT / "Table3_win_loss_PC_vs_NFBASE.csv",
    index=False,
)


print("\nTABLE 3")
print(win_table.to_string(index=False))


# =========================
# Save clean LaTeX tables
# =========================

summary.to_latex(
    OUT / "Table1_performance_ranking.tex",
    index=False,
    float_format="%.4f",
)


win_table.to_latex(
    OUT / "Table3_win_loss_PC_vs_NFBASE.tex",
    index=False,
    float_format="%.2f",
)


print("\nSaved to:", OUT)