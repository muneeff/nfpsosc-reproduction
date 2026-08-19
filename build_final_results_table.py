from pathlib import Path
import json
import pandas as pd


RESULTS_DIR = Path(
    "outputs/v2/final_external_memory_safe/results"
)

OUTPUT = Path(
    "outputs/v2/final_external_memory_safe/analysis/final_results_master.csv"
)


rows = []

for file in RESULTS_DIR.glob("*.json"):

    with file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("status") != "success":
        continue

    task = data.get("task", {})
    metrics = data.get("metrics", {})
    config = data.get("selected_config", {})

    objective = config.get(
        "objective_components",
        {}
    )

    rows.append({

        # run identity
        "run_id":
            task.get("run_id"),

        # task information
        "domain":
            task.get("domain"),

        "source":
            task.get("source"),

        "frequency":
            task.get("frequency"),

        "series_id":
            task.get("series_id"),

        "method":
            task.get("method"),

        "optimizer_seed":
            task.get("optimizer_seed"),

        "dgp_seed":
            task.get("dgp_seed"),
        "generator":
            task.get("generator"),
        "noise_level":
            task.get("noise_level"),

        "n_lags":
            task.get("n_lags"),

        "seasonal_period":
            task.get("seasonal_period"),

        "alpha":
            task.get("alpha"),

        "radius":
            task.get("radius"),


        # metrics
        "rmse":
            metrics.get("rmse"),

        "mae":
            metrics.get("mae"),

        "smape":
            metrics.get("smape"),

        "mase":
            metrics.get("mase"),

        "rmsse":
            metrics.get("rmsse"),


        # PC-NFPSO configuration
        "boundary":
            config.get("boundary"),

        "dynamics":
            config.get("dynamics"),

        "n_rules":
            data.get("n_rules"),


        # objective
        "fit_rmse":
            objective.get("fit_rmse"),

        "validation_rmse":
            objective.get("validation_rmse"),

        "sensitivity_penalty":
            objective.get("sensitivity_penalty"),

        "objective_total":
            objective.get("total"),

    })


df = pd.DataFrame(rows)

df = df.sort_values(
    [
        "source",
        "frequency",
        "series_id",
        "method"
    ]
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False
)

print("ROWS =", len(df))
print("OUTPUT =", OUTPUT)
print(df.head())