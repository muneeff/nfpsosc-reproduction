from pathlib import Path
import argparse
import sys
import time
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from run_real_data_benchmarks import run_baseline_model, run_nfpso_model


OUT_DIR = ROOT / "outputs" / "real_short_series"
SELECTED_CSV = OUT_DIR / "selected_real_series.csv"
NFPSO_INPUT_DIR = OUT_DIR / "nfpso_inputs"

BASELINE_MODELS = [
    "naive_1",
    "seasonal_naive",
    "drift",
    "arima",
    "ets",
    "theta",
    "svr",
    "xgboost",
]

DISPLAY_MODEL_ORDER = BASELINE_MODELS + ["projected_constricted_nfpso"]


def frequency_to_season_length(freq: str) -> int:
    freq = str(freq).lower()
    if "monthly" in freq:
        return 12
    if "quarter" in freq:
        return 4
    return 1


def frequency_to_pandas_freq(freq: str) -> str:
    freq = str(freq).lower()
    if "monthly" in freq:
        return "MS"
    if "quarter" in freq:
        return "QS"
    return "YS"


def read_series(row: pd.Series) -> np.ndarray:
    path = ROOT / str(row["file_path"])
    if not path.exists():
        raise FileNotFoundError(f"Missing series file: {path}")

    df = pd.read_csv(path)

    if "revenue_million_yer" in df.columns:
        values = pd.to_numeric(df["revenue_million_yer"], errors="coerce")
    elif "value" in df.columns:
        values = pd.to_numeric(df["value"], errors="coerce")
    else:
        raise ValueError(f"No value column found in {path}. Columns: {list(df.columns)}")

    values = values.dropna().to_numpy(dtype=float)

    if len(values) < 30:
        raise ValueError(f"Series too short after cleaning: {row['series_id']} length={len(values)}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite values found in series: {row['series_id']}")

    return values


def make_nfpso_input(row: pd.Series, values: np.ndarray) -> Path:
    """
    NFPSO currently expects a CSV compatible with load_revenue_series:
    date,revenue_million_yer

    We create a temporary compatible file for public benchmark series.
    For Yemen tax revenue, the original file is already compatible.
    """
    source_path = ROOT / str(row["file_path"])

    if str(row["series_id"]) == "yemen_tax_revenue":
        return source_path

    NFPSO_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    freq = frequency_to_pandas_freq(str(row["frequency"]))
    dates = pd.date_range(start="2000-01-01", periods=len(values), freq=freq)

    out_path = NFPSO_INPUT_DIR / f"{row['series_id']}.csv"
    pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "revenue_million_yer": values,
    }).to_csv(out_path, index=False, encoding="utf-8")

    return out_path


def metric_row_base(row: pd.Series) -> Dict[str, Any]:
    return {
        "dataset": row["series_id"],
        "source_dataset": row["source"],
        "frequency": row["frequency"],
        "series_length": int(row["length"]),
    }


def aggregate_nfpso_rows(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ok_rows = [
        r for r in rows
        if str(r.get("status", "ok")).lower() == "ok"
    ]

    if not ok_rows:
        if rows:
            first = dict(rows[0])
            first["status"] = "failed"
            first["error"] = "All PC-NFPSO seeds failed."
            first["seed"] = "aggregate"
            return first
        return None

    df = pd.DataFrame(ok_rows)

    first = dict(ok_rows[0])
    first["seed"] = "mean_over_seeds"
    first["n_seeds"] = len(ok_rows)
    first["status"] = "ok"
    first["error"] = None
    first["source"] = "nfpso_runner_seed_mean"

    metric_cols = [
        "rmse", "mae", "mase",
        "smape_percent", "mape_percent", "r2",
        "error_mean", "error_std_sample",
        "test_rmse", "test_mae", "test_mase",
        "test_smape_percent", "test_mape_percent", "test_r2",
        "runtime_seconds",
    ]

    for col in metric_cols:
        if col in df.columns:
            first[col] = pd.to_numeric(df[col], errors="coerce").mean()

    return first


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real short-series benchmarks for Yemen + selected M4/M3 series."
    )
    parser.add_argument("--quick", action="store_true", help="Fast smoke test: fewer models and tiny NFPSO budget.")
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--n-lags", type=int, default=5)
    parser.add_argument("--radius", type=float, default=1.00)
    parser.add_argument("--particles", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    args = parser.parse_args()

    if not SELECTED_CSV.exists():
        raise FileNotFoundError(
            f"Missing {SELECTED_CSV}. Run first:\n"
            "py select_real_short_series.py"
        )

    selected = pd.read_csv(SELECTED_CSV)

    if len(selected) != 6:
        raise RuntimeError(f"Expected 6 selected series, found {len(selected)}")

    if args.quick:
        baselines = ["naive_1", "seasonal_naive", "drift"]
        seeds = [1]
        particles = 5
        iterations = 5
    else:
        baselines = BASELINE_MODELS
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        particles = int(args.particles)
        iterations = int(args.iterations)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows: List[Dict[str, Any]] = []
    seed_metric_rows: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []
    failed_rows: List[Dict[str, Any]] = []

    config = {
        "nfpso": {
            "sensitivity_weight": 0.001,
            "validation_fraction": 0.20,
            "activation_floor": 1e-4,
            "activation_penalty_weight": 0.01,
        }
    }

    for _, row in selected.iterrows():
        dataset = str(row["series_id"])
        frequency = str(row["frequency"])
        season_length = frequency_to_season_length(frequency)

        print(f"\n[DATASET] {dataset} frequency={frequency} season_length={season_length}")

        values = read_series(row)
        nfpso_csv = make_nfpso_input(row, values)

        # Baselines
        for model in baselines:
            base_info = metric_row_base(row)
            try:
                mrow, pred = run_baseline_model(
                    series=values,
                    dataset_name=dataset,
                    model=model,
                    n_lags=int(args.n_lags),
                    season_length=season_length,
                    train_fraction=float(args.train_fraction),
                    initial_train_size=None,
                )
                mrow.update(base_info)
                metric_rows.append(mrow)
                if pred is not None:
                    pred["frequency"] = frequency
                    pred["series_length"] = int(row["length"])
                    prediction_frames.append(pred)

                if str(mrow.get("status", "ok")).lower() != "ok":
                    failed_rows.append(mrow)

            except Exception as exc:
                failed = {
                    **base_info,
                    "dataset": dataset,
                    "dataset_type": "real",
                    "model": model,
                    "source": "baseline_runner",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                metric_rows.append(failed)
                failed_rows.append(failed)

        # PC-NFPSO, one row per seed, then aggregate row
        nfpso_seed_rows: List[Dict[str, Any]] = []

        for seed in seeds:
            base_info = metric_row_base(row)
            try:
                nrow, pred = run_nfpso_model(
                    csv_path=nfpso_csv,
                    output_root=OUT_DIR / "nfpso_runs" / dataset,
                    dataset_name=dataset,
                    protocol="research_baseline",
                    objective="composite",
                    radius=float(args.radius),
                    seed=int(seed),
                    particles=particles,
                    iterations=iterations,
                    config=config,
                    n_lags=int(args.n_lags),
                    train_fraction=float(args.train_fraction),
                )
                nrow.update(base_info)
                nfpso_seed_rows.append(nrow)
                seed_metric_rows.append(nrow)

                if pred is not None:
                    pred["frequency"] = frequency
                    pred["series_length"] = int(row["length"])
                    pred["seed"] = seed
                    prediction_frames.append(pred)

                if str(nrow.get("status", "ok")).lower() != "ok":
                    failed_rows.append(nrow)

            except Exception as exc:
                failed = {
                    **base_info,
                    "dataset": dataset,
                    "dataset_type": "real",
                    "model": "projected_constricted_nfpso",
                    "source": "nfpso_runner",
                    "protocol": "research_baseline",
                    "objective": "composite",
                    "sc_radius": float(args.radius),
                    "seed": int(seed),
                    "particles": particles,
                    "iterations": iterations,
                    "n_lags": int(args.n_lags),
                    "train_fraction": float(args.train_fraction),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                nfpso_seed_rows.append(failed)
                seed_metric_rows.append(failed)
                failed_rows.append(failed)

        agg = aggregate_nfpso_rows(nfpso_seed_rows)
        if agg is not None:
            metric_rows.append(agg)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_by_seed_df = pd.DataFrame(seed_metric_rows)

    metrics_path = OUT_DIR / "real_short_series_metrics.csv"
    seed_metrics_path = OUT_DIR / "real_short_series_metrics_by_seed.csv"
    predictions_path = OUT_DIR / "real_short_series_predictions.csv"
    manifest_path = OUT_DIR / "manifest.csv"
    failed_path = OUT_DIR / "failed_model_runs.csv"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    metrics_by_seed_df.to_csv(seed_metrics_path, index=False, encoding="utf-8-sig")

    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True, sort=False).to_csv(
            predictions_path,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        pd.DataFrame().to_csv(predictions_path, index=False, encoding="utf-8-sig")

    manifest_cols = [
        c for c in [
            "dataset", "source_dataset", "frequency", "series_length",
            "model", "source", "protocol", "objective", "sc_radius",
            "seed", "n_seeds", "particles", "iterations", "n_lags",
            "season_length", "train_fraction", "runtime_seconds",
            "status", "error", "run_dir",
        ]
        if c in metrics_df.columns
    ]
    metrics_df[manifest_cols].to_csv(manifest_path, index=False, encoding="utf-8-sig")

    failed_df = pd.DataFrame(failed_rows)
    failed_df.to_csv(failed_path, index=False, encoding="utf-8-sig")

    print("\n[DONE]")
    print(f"metrics:         {metrics_path}")
    print(f"metrics_by_seed: {seed_metrics_path}")
    print(f"predictions:     {predictions_path}")
    print(f"manifest:        {manifest_path}")
    print(f"failed:          {failed_path}")
    print(f"rows:            {len(metrics_df)}")
    print(f"failed rows:     {len(failed_df)}")


if __name__ == "__main__":
    main()