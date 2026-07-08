"""
Run leakage-aware forecasting baselines for the SS-NFPSO Q1 pipeline.

This script is intentionally conservative:
- it uses chronological one-step-ahead evaluation;
- it skips missing private real datasets instead of failing;
- it continues when optional dependencies such as statsmodels or xgboost are
  not installed;
- it writes outputs under outputs/, which should remain ignored by Git.

Examples
--------
Quick synthetic smoke test:

    py run_baselines.py --quick --models naive_1,drift

Minimal configured run:

    py run_baselines.py --config configs/q1_minimal_experiments.json

Use only simple baselines:

    py run_baselines.py --models naive_1,seasonal_naive,drift
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
    """Parse a comma-separated string into a list."""

    if value is None or value.strip() == "":
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def ensure_dir(path: Path) -> Path:
    """Create directory and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def find_dataset_spec(datasets_config: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    """Find dataset specification by name."""

    for spec in datasets_config.get("datasets", []):
        if spec.get("name") == name:
            return spec
    return None


def load_real_series(spec: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Load a real dataset if its local file exists."""

    path = ROOT / spec["path"]
    if not path.exists():
        print(f"[SKIP] real dataset not found: {spec.get('name')} -> {path}")
        return None

    df = pd.read_csv(path)
    target_col = spec.get("target_column", "value")
    date_col = spec.get("date_column", "date")

    if target_col not in df.columns:
        print(f"[SKIP] target column '{target_col}' not found in {path}")
        return None

    out = pd.DataFrame(
        {
            "date": df[date_col] if date_col in df.columns else range(len(df)),
            "value": pd.to_numeric(df[target_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["value"]).reset_index(drop=True)
    if len(out) < 20:
        print(f"[SKIP] dataset too short after cleaning: {spec.get('name')}")
        return None
    return out


def build_real_jobs(
    exp_config: Dict[str, Any],
    datasets_config: Dict[str, Any],
    only_synthetic: bool,
) -> List[Dict[str, Any]]:
    """Build jobs for real datasets."""

    if only_synthetic:
        return []

    jobs: List[Dict[str, Any]] = []
    include_real = exp_config.get("datasets", {}).get("include_real", [])
    for name in include_real:
        spec = find_dataset_spec(datasets_config, name)
        if spec is None:
            print(f"[SKIP] real dataset spec not found: {name}")
            continue

        df = load_real_series(spec)
        if df is None:
            continue

        jobs.append(
            {
                "dataset": name,
                "dataset_type": "real",
                "series": df["value"].to_numpy(dtype=float),
                "n_lags": int(spec.get("n_lags", exp_config.get("nfpso", {}).get("default_n_lags", 5))),
                "season_length": int(spec.get("season_length", 1)),
                "train_fraction": float(spec.get("train_fraction", 0.8)),
                "metadata": {
                    "frequency": spec.get("frequency"),
                    "role": spec.get("role"),
                    "path": spec.get("path"),
                },
            }
        )
    return jobs


def build_synthetic_jobs(
    exp_config: Dict[str, Any],
    datasets_config: Dict[str, Any],
    quick: bool,
) -> List[Dict[str, Any]]:
    """Build jobs for synthetic datasets."""

    dataset_block = exp_config.get("datasets", {})
    synthetic_names = list(dataset_block.get("include_synthetic", []))
    lengths = list(dataset_block.get("synthetic_lengths", [180]))
    noise_levels = list(dataset_block.get("noise_levels", [0.05]))
    seeds = list(exp_config.get("random_seeds", [1]))

    if quick:
        synthetic_names = synthetic_names[:2] or ["synthetic_linear_ar"]
        lengths = lengths[:1] or [60]
        lengths = [min(int(lengths[0]), 60)]
        noise_levels = noise_levels[:1] or [0.05]
        seeds = seeds[:1] or [1]

    jobs: List[Dict[str, Any]] = []
    for name in synthetic_names:
        spec = find_dataset_spec(datasets_config, name) or {}
        generator = spec.get("generator", name)
        season_length = int(spec.get("season_length", 12 if "seasonal" in name else 1))
        n_lags = int(spec.get("n_lags", 12 if "seasonal" in name else 5))

        for length in lengths:
            for noise_level in noise_levels:
                for seed in seeds:
                    df = make_synthetic_series(
                        generator,
                        length=int(length),
                        noise_level=float(noise_level),
                        seed=int(seed),
                        season_length=season_length if "seasonal" in generator else None,
                    )
                    jobs.append(
                        {
                            "dataset": f"{name}_n{length}_noise{noise_level}_seed{seed}",
                            "dataset_type": "synthetic",
                            "series": df["value"].to_numpy(dtype=float),
                            "n_lags": n_lags,
                            "season_length": season_length,
                            "train_fraction": float(spec.get("train_fraction", 0.8)),
                            "metadata": {
                                "generator": generator,
                                "length": int(length),
                                "noise_level": float(noise_level),
                                "seed": int(seed),
                            },
                        }
                    )
    return jobs


def normalize_model_list(exp_config: Dict[str, Any], cli_models: Optional[List[str]], quick: bool) -> List[str]:
    """Resolve model list."""

    models = cli_models or list(exp_config.get("models", ["naive_1", "seasonal_naive", "drift"]))
    if quick:
        # Keep quick tests light and dependency-free unless user explicitly asked.
        if cli_models is None:
            models = ["naive_1", "seasonal_naive", "drift"]
    return models


def run_one_job(
    job: Dict[str, Any],
    model: str,
    initial_train_size: Optional[int],
) -> tuple[Optional[BaselineResult], Dict[str, Any]]:
    """Run one dataset-model job and return result plus row metadata."""

    start = time.perf_counter()
    error_message = None
    result: Optional[BaselineResult] = None

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
            job["series"],
            model=model,
            initial_train_size=initial_train_size,
            train_fraction=job["train_fraction"] if initial_train_size is None else None,
            n_lags=job["n_lags"],
            season_length=job["season_length"],
            **kwargs,
        )
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"

    runtime = time.perf_counter() - start
    row_meta = {
        "dataset": job["dataset"],
        "dataset_type": job["dataset_type"],
        "model": model,
        "n_obs": len(job["series"]),
        "n_lags": job["n_lags"],
        "season_length": job["season_length"],
        "train_fraction": job["train_fraction"],
        "runtime_seconds": runtime,
        "status": "ok" if result is not None else "failed",
        "error": error_message,
        **{f"meta_{k}": v for k, v in job.get("metadata", {}).items()},
    }
    return result, row_meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Run leakage-aware baseline forecasts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Experiment config JSON.")
    parser.add_argument("--datasets-config", default=str(DEFAULT_DATASETS_CONFIG), help="Dataset config JSON.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--models", default=None, help="Comma-separated model list.")
    parser.add_argument("--quick", action="store_true", help="Run a small smoke test.")
    parser.add_argument("--only-synthetic", action="store_true", help="Skip real datasets.")
    parser.add_argument("--initial-train-size", type=int, default=None, help="Optional fixed initial train size.")
    args = parser.parse_args()

    exp_config = load_json(Path(args.config))
    datasets_config = load_json(Path(args.datasets_config))

    output_root = Path(args.output_dir or exp_config.get("output_dir", "outputs/q1_minimal"))
    output_dir = ensure_dir(ROOT / output_root / "baselines")

    cli_models = parse_csv_list(args.models)
    models = normalize_model_list(exp_config, cli_models=cli_models, quick=args.quick)

    jobs = []
    jobs.extend(build_real_jobs(exp_config, datasets_config, only_synthetic=args.only_synthetic))
    jobs.extend(build_synthetic_jobs(exp_config, datasets_config, quick=args.quick))

    if not jobs:
        print("[ERROR] No dataset jobs were created.")
        return 2

    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] jobs={len(jobs)} models={models}")

    metric_rows: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []

    for job_idx, job in enumerate(jobs, start=1):
        for model in models:
            print(f"[RUN] {job_idx}/{len(jobs)} dataset={job['dataset']} model={model}")
            result, row_meta = run_one_job(job, model, args.initial_train_size)

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

    if prediction_frames:
        predictions_df = pd.concat(prediction_frames, ignore_index=True)
    else:
        predictions_df = pd.DataFrame()
    predictions_path = output_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    print(f"[DONE] metrics: {metrics_path}")
    print(f"[DONE] predictions: {predictions_path}")
    print("[NOTE] outputs/ is ignored by Git and should not be committed if it contains real data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
