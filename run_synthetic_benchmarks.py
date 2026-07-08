"""
Run synthetic benchmark experiments for the SS-NFPSO Q1 pipeline.

This script generates controlled synthetic time series and evaluates selected
forecasting baselines under leakage-aware rolling one-step-ahead evaluation.

The goal is not to prove real-world superiority. The goal is to stress-test
forecasting behavior under controlled mechanisms such as linear dynamics,
nonlinear smoothness, regime switching, outbreak-like spikes, and seasonality.

Examples
--------
Quick smoke test:

    py run_synthetic_benchmarks.py --quick --models naive_1,drift

Minimal configured run:

    py run_synthetic_benchmarks.py --config configs/q1_minimal_experiments.json

More baselines, if dependencies are installed:

    py run_synthetic_benchmarks.py --quick --models naive_1,drift,arima,ets,svr
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nfpsosc.baselines import BaselineResult, run_baseline
from nfpsosc.synthetic import make_synthetic_series


DEFAULT_CONFIG = ROOT / "configs" / "q1_minimal_experiments.json"
DEFAULT_DATASETS_CONFIG = ROOT / "configs" / "datasets.json"


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated CLI values."""

    if value is None or value.strip() == "":
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def ensure_dir(path: Path) -> Path:
    """Create output directory if needed."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def find_dataset_spec(datasets_config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Return a dataset spec by name or an empty dictionary."""

    for spec in datasets_config.get("datasets", []):
        if spec.get("name") == name:
            return spec
    return {}


def resolve_synthetic_plan(
    exp_config: Dict[str, Any],
    datasets_config: Dict[str, Any],
    quick: bool,
) -> List[Dict[str, Any]]:
    """Build synthetic benchmark jobs from configuration."""

    dataset_block = exp_config.get("datasets", {})
    names = list(dataset_block.get("include_synthetic", []))
    lengths = list(dataset_block.get("synthetic_lengths", [180]))
    noise_levels = list(dataset_block.get("noise_levels", [0.05]))
    seeds = list(exp_config.get("random_seeds", [1]))

    if quick:
        names = names[:2] or ["synthetic_linear_ar", "synthetic_nonlinear_sine"]
        lengths = [min(int(lengths[0]) if lengths else 180, 60)]
        noise_levels = noise_levels[:2] or [0.0, 0.05]
        seeds = seeds[:1] or [1]

    jobs: List[Dict[str, Any]] = []
    for name in names:
        spec = find_dataset_spec(datasets_config, name)
        generator = spec.get("generator", name)
        n_lags = int(spec.get("n_lags", 12 if "seasonal" in generator else 5))
        season_length = int(spec.get("season_length", 12 if "seasonal" in generator else 1))
        train_fraction = float(spec.get("train_fraction", 0.8))

        for length in lengths:
            for noise_level in noise_levels:
                for seed in seeds:
                    jobs.append(
                        {
                            "dataset": f"{name}_n{length}_noise{noise_level}_seed{seed}",
                            "base_name": name,
                            "generator": generator,
                            "length": int(length),
                            "noise_level": float(noise_level),
                            "seed": int(seed),
                            "n_lags": n_lags,
                            "season_length": season_length,
                            "train_fraction": train_fraction,
                        }
                    )
    return jobs


def resolve_models(exp_config: Dict[str, Any], cli_models: Optional[List[str]], quick: bool) -> List[str]:
    """Resolve model names for this script."""

    if cli_models is not None:
        return cli_models

    models = list(exp_config.get("models", ["naive_1", "seasonal_naive", "drift"]))
    # Synthetic smoke tests should be light and dependency-free by default.
    if quick:
        return ["naive_1", "seasonal_naive", "drift"]

    # Keep only baseline models for this script; NFPSO has its own experiment path.
    excluded = {"legacy_nfpso", "projected_constricted_nfpso"}
    return [m for m in models if m not in excluded]


def run_one(
    job: Dict[str, Any],
    model: str,
    initial_train_size: Optional[int] = None,
) -> tuple[Optional[BaselineResult], Dict[str, Any]]:
    """Generate one synthetic series and run one model."""

    start_time = time.perf_counter()
    error_message = None
    result: Optional[BaselineResult] = None

    df = make_synthetic_series(
        job["generator"],
        length=job["length"],
        noise_level=job["noise_level"],
        seed=job["seed"],
        season_length=job["season_length"] if "seasonal" in job["generator"] else None,
    )
    series = df["value"].to_numpy(dtype=float)

    try:
        kwargs: Dict[str, Any] = {}
        if model == "arima":
            kwargs["order"] = (1, 1, 0)
        elif model == "ets":
            kwargs["trend"] = "add"
            kwargs["seasonal"] = "add" if job["season_length"] > 1 else None
        elif model == "svr":
            kwargs.update({"C": 10.0, "epsilon": 0.05, "gamma": "scale"})
        elif model == "xgboost":
            kwargs.update({"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05})

        result = run_baseline(
            series,
            model=model,
            initial_train_size=initial_train_size,
            train_fraction=job["train_fraction"] if initial_train_size is None else None,
            n_lags=job["n_lags"],
            season_length=job["season_length"],
            **kwargs,
        )
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"

    runtime = time.perf_counter() - start_time
    row_meta = {
        "dataset": job["dataset"],
        "dataset_type": "synthetic",
        "base_name": job["base_name"],
        "generator": job["generator"],
        "length": job["length"],
        "noise_level": job["noise_level"],
        "seed": job["seed"],
        "model": model,
        "n_lags": job["n_lags"],
        "season_length": job["season_length"],
        "train_fraction": job["train_fraction"],
        "runtime_seconds": runtime,
        "status": "ok" if result is not None else "failed",
        "error": error_message,
    }

    return result, row_meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Run synthetic benchmark baselines.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Experiment config JSON.")
    parser.add_argument("--datasets-config", default=str(DEFAULT_DATASETS_CONFIG), help="Dataset config JSON.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--models", default=None, help="Comma-separated model names.")
    parser.add_argument("--quick", action="store_true", help="Run a small smoke test.")
    parser.add_argument("--initial-train-size", type=int, default=None, help="Optional fixed initial train size.")
    args = parser.parse_args()

    exp_config = load_json(Path(args.config))
    datasets_config = load_json(Path(args.datasets_config))

    output_root = Path(args.output_dir or exp_config.get("output_dir", "outputs/q1_minimal"))
    output_dir = ensure_dir(ROOT / output_root / "synthetic_benchmarks")

    jobs = resolve_synthetic_plan(exp_config, datasets_config, quick=args.quick)
    models = resolve_models(exp_config, parse_csv_list(args.models), quick=args.quick)

    if not jobs:
        print("[ERROR] No synthetic jobs were created.")
        return 2

    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] synthetic_jobs={len(jobs)} models={models}")

    metric_rows: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []

    for job_idx, job in enumerate(jobs, start=1):
        for model in models:
            print(f"[RUN] {job_idx}/{len(jobs)} dataset={job['dataset']} model={model}")
            result, row_meta = run_one(job, model, initial_train_size=args.initial_train_size)

            if result is not None:
                row = {**row_meta, **result.metrics}
                row["failed_steps"] = len(result.failed_steps)
                row["n_test"] = len(result.y_true)
                metric_rows.append(row)

                pred_df = result.to_frame()
                for key, value in row_meta.items():
                    pred_df[key] = value
                prediction_frames.append(pred_df)
            else:
                metric_rows.append(row_meta)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    predictions_path = output_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    summary_path = output_dir / "summary_by_generator.csv"
    if not metrics_df.empty and "rmse" in metrics_df.columns:
        summary_cols = [
            "base_name",
            "generator",
            "noise_level",
            "model",
            "rmse",
            "mae",
            "smape_percent",
            "mase",
            "runtime_seconds",
        ]
        available_cols = [c for c in summary_cols if c in metrics_df.columns]
        summary = (
            metrics_df[metrics_df["status"] == "ok"][available_cols]
            .groupby(["base_name", "generator", "noise_level", "model"], as_index=False)
            .mean(numeric_only=True)
        )
        summary.to_csv(summary_path, index=False)
    else:
        pd.DataFrame().to_csv(summary_path, index=False)

    print(f"[DONE] metrics: {metrics_path}")
    print(f"[DONE] predictions: {predictions_path}")
    print(f"[DONE] summary: {summary_path}")
    print("[NOTE] outputs/ is ignored by Git and should not be committed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
