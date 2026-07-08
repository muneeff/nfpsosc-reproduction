"""
Run NFPSO experiments on controlled synthetic time series.

This script closes an important gap in the Q1 pipeline:
`run_synthetic_benchmarks.py` evaluates only baseline models on synthetic data,
while this script evaluates NFPSO on the same synthetic conditions.

Main outputs
------------
    outputs/q1_minimal/synthetic_nfpso_experiments/metrics.csv
    outputs/q1_minimal/synthetic_nfpso_experiments/predictions.csv
    outputs/q1_minimal/synthetic_nfpso_experiments/manifest.csv

Optional merged outputs
-----------------------
When `--merge-with-baselines` is enabled, the script also creates:

    outputs/q1_minimal/synthetic_unified_benchmarks/metrics.csv
    outputs/q1_minimal/synthetic_unified_benchmarks/predictions.csv

This merged table is useful for:

    py run_statistical_tests.py --inputs outputs/q1_minimal/synthetic_unified_benchmarks/metrics.csv --reference-model projected_constricted_nfpso --quick

Important scientific note
-------------------------
Synthetic experiments support controlled mechanism-level claims. They do not
replace real-data validation. Interpret them as stress tests, not as proof of
universal superiority.

Examples
--------
Quick smoke test:

    py run_synthetic_nfpso_experiments.py --quick

Quick run and merge with existing synthetic baseline metrics:

    py run_synthetic_nfpso_experiments.py --quick --merge-with-baselines

Run both NFPSO protocols:

    py run_synthetic_nfpso_experiments.py --protocols research_baseline,legacy_intent

Custom synthetic settings:

    py run_synthetic_nfpso_experiments.py --datasets synthetic_linear_ar,synthetic_nonlinear_sine --lengths 120 --noise-levels 0.0,0.05 --seeds 1,2,3
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

from nfpsosc.experiment import ExperimentConfig, run_experiment
from nfpsosc.synthetic import make_synthetic_series


DEFAULT_CONFIG = ROOT / "configs" / "q1_minimal_experiments.json"
DEFAULT_DATASETS_CONFIG = ROOT / "configs" / "datasets.json"
DEFAULT_SYNTHETIC_BASELINE_METRICS = ROOT / "outputs" / "q1_minimal" / "synthetic_benchmarks" / "metrics.csv"
DEFAULT_SYNTHETIC_BASELINE_PREDICTIONS = ROOT / "outputs" / "q1_minimal" / "synthetic_benchmarks" / "predictions.csv"


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file if present."""

    if not path.exists():
        print(f"[WARN] config not found, using defaults: {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    """Create and return a directory."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated strings."""

    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_list(value: Optional[str]) -> Optional[List[float]]:
    """Parse comma-separated floats."""

    items = parse_csv_list(value)
    if items is None:
        return None
    return [float(item) for item in items]


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated integers."""

    items = parse_csv_list(value)
    if items is None:
        return None
    return [int(item) for item in items]


def relpath(path: Path) -> str:
    """Return path relative to repository root when possible."""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def find_dataset_spec(datasets_config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Find a dataset specification by name."""

    for spec in datasets_config.get("datasets", []):
        if spec.get("name") == name:
            return spec
    return {}


def resolve_protocols(cli_protocols: Optional[List[str]], quick: bool) -> List[str]:
    """Resolve NFPSO protocols."""

    protocols = cli_protocols or ["research_baseline"]
    valid = {"legacy_intent", "research_baseline"}
    invalid = [p for p in protocols if p not in valid]
    if invalid:
        raise ValueError(f"Invalid protocols: {invalid}. Valid protocols: {sorted(valid)}")

    if quick:
        return ["research_baseline"] if "research_baseline" in protocols else protocols[:1]
    return protocols


def resolve_synthetic_jobs(
    exp_config: Dict[str, Any],
    datasets_config: Dict[str, Any],
    cli_datasets: Optional[List[str]],
    cli_lengths: Optional[List[int]],
    cli_noise_levels: Optional[List[float]],
    cli_seeds: Optional[List[int]],
    quick: bool,
) -> List[Dict[str, Any]]:
    """Resolve synthetic series jobs matching run_synthetic_benchmarks.py naming."""

    dataset_block = exp_config.get("datasets", {})
    names = cli_datasets or list(dataset_block.get("include_synthetic", []))
    lengths = cli_lengths or list(dataset_block.get("synthetic_lengths", [180]))
    noise_levels = cli_noise_levels or list(dataset_block.get("noise_levels", [0.05]))
    seeds = cli_seeds or list(exp_config.get("random_seeds", [1]))

    if quick:
        names = names[:2] or ["synthetic_linear_ar", "synthetic_nonlinear_sine"]
        lengths = [min(int(lengths[0]) if lengths else 180, 60)]
        noise_levels = noise_levels[:2] or [0.0, 0.05]
        seeds = seeds[:1] or [1]

    jobs: List[Dict[str, Any]] = []
    for name in names:
        spec = find_dataset_spec(datasets_config, name)
        generator = spec.get("generator", name)
        n_lags = int(spec.get("n_lags", exp_config.get("nfpso", {}).get("default_n_lags", 5)))
        season_length = int(spec.get("season_length", 12 if "seasonal" in generator else 1))
        train_fraction = float(spec.get("train_fraction", exp_config.get("train_fraction", 0.85)))

        for length in lengths:
            for noise_level in noise_levels:
                for seed in seeds:
                    jobs.append(
                        {
                            "dataset": f"{name}_n{int(length)}_noise{float(noise_level)}_seed{int(seed)}",
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


def resolve_radii(
    exp_config: Dict[str, Any],
    cli_radii: Optional[List[float]],
    quick: bool,
) -> List[float]:
    """Resolve NFPSO SC radii."""

    radii = (
        cli_radii
        or exp_config.get("radius_sweep", {}).get("radii")
        or exp_config.get("stability_sweep", {}).get("radii")
        or exp_config.get("nfpso", {}).get("radii")
        or [0.55]
    )
    radii = [float(r) for r in radii]

    if quick:
        return [0.55 if 0.55 in radii else radii[0]]
    return radii


def resolve_particles_iterations(
    exp_config: Dict[str, Any],
    cli_particles: Optional[int],
    cli_iterations: Optional[int],
    quick: bool,
) -> tuple[int, int]:
    """Resolve PSO budget."""

    nfpso_cfg = exp_config.get("nfpso", {})
    particles = int(cli_particles if cli_particles is not None else nfpso_cfg.get("particles", 12))
    iterations = int(cli_iterations if cli_iterations is not None else nfpso_cfg.get("iterations", 40))

    if quick:
        particles = min(particles, 5)
        iterations = min(iterations, 5)

    return particles, iterations


def objective_for_protocol(protocol: str, objective: str) -> str:
    """Resolve objective."""

    if objective != "auto":
        return objective
    if protocol == "legacy_intent":
        return "error_std"
    return "composite"


def model_name_for_protocol(protocol: str) -> str:
    """Map protocol to model name."""

    if protocol == "legacy_intent":
        return "legacy_nfpso"
    if protocol == "research_baseline":
        return "projected_constricted_nfpso"
    return protocol


def write_synthetic_csv(job: Dict[str, Any], output_dir: Path) -> Path:
    """
    Generate one synthetic series and write it using the real-data CSV schema.

    `run_experiment` expects a revenue-style CSV, so we provide both:
    - value
    - revenue_million_yer

    This avoids touching private data and keeps synthetic inputs under outputs/.
    """

    df = make_synthetic_series(
        job["generator"],
        length=job["length"],
        noise_level=job["noise_level"],
        seed=job["seed"],
        season_length=job["season_length"] if "seasonal" in job["generator"] else None,
    )

    out = pd.DataFrame(
        {
            "date": df["date"] if "date" in df.columns else pd.date_range("2000-01-01", periods=len(df), freq="MS"),
            "value": pd.to_numeric(df["value"], errors="coerce"),
        }
    )
    out["revenue_million_yer"] = out["value"]

    path = output_dir / f"{job['dataset']}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def make_experiment_config(
    csv_path: Path,
    output_dir: Path,
    protocol: str,
    objective: str,
    radius: float,
    seed: int,
    particles: int,
    iterations: int,
    n_lags: int,
    train_fraction: float,
    exp_config: Dict[str, Any],
) -> ExperimentConfig:
    """Create ExperimentConfig for one synthetic NFPSO run."""

    nfpso_cfg = exp_config.get("nfpso", {})

    return ExperimentConfig(
        csv_path=str(csv_path),
        output_dir=str(output_dir),
        n_lags=int(n_lags),
        train_fraction=float(train_fraction),
        sc_radius=float(radius),
        protocol=protocol,
        objective=objective,
        sensitivity_weight=float(nfpso_cfg.get("sensitivity_weight", 0.001)),
        validation_fraction=float(nfpso_cfg.get("validation_fraction", 0.20)),
        activation_floor=float(nfpso_cfg.get("activation_floor", 1e-4)),
        activation_penalty_weight=float(nfpso_cfg.get("activation_penalty_weight", 0.01)),
        seed=int(seed),
        iterations=int(iterations),
        particles=int(particles),
    )


def flatten(prefix: str, values: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten dictionary with a prefix."""

    if not values:
        return {}
    return {
        f"{prefix}_{key}": value
        for key, value in values.items()
        if isinstance(value, (int, float, str, bool)) or value is None
    }


def copy_test_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Copy test metrics to unprefixed columns expected by statistical scripts."""

    metrics = summary.get("test_metrics") or {}
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) or value is None
    }


def summarize_history(history_path: Path) -> Dict[str, Any]:
    """Summarize optimizer history if available."""

    if not history_path.exists():
        return {}

    try:
        hist = pd.read_csv(history_path)
    except Exception:
        return {}

    out: Dict[str, Any] = {"history_rows": int(len(hist))}
    if hist.empty:
        return out

    for col in [
        "best_cost",
        "mean_velocity_norm",
        "swarm_diameter",
        "best_parameter_drift",
        "boundary_hits",
    ]:
        if col in hist.columns:
            numeric = pd.to_numeric(hist[col], errors="coerce").dropna()
            if not numeric.empty:
                out[f"history_initial_{col}"] = float(numeric.iloc[0])
                out[f"history_final_{col}"] = float(numeric.iloc[-1])
                out[f"history_min_{col}"] = float(numeric.min())
                out[f"history_max_{col}"] = float(numeric.max())

    if "best_cost" in hist.columns:
        numeric = pd.to_numeric(hist["best_cost"], errors="coerce").dropna()
        if len(numeric) >= 2:
            out["history_best_cost_improvement"] = float(numeric.iloc[0] - numeric.iloc[-1])

    return out


def read_predictions_if_available(run_dir: Path, row_meta: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Read test predictions written by run_experiment, if available."""

    candidates = [
        run_dir / "test_predictions.csv",
        run_dir / "predictions.csv",
    ]
    pred_path = next((p for p in candidates if p.exists()), None)
    if pred_path is None:
        return None

    try:
        pred = pd.read_csv(pred_path)
    except Exception:
        return None

    if pred.empty:
        return None

    pred = pred.copy()
    for key, value in row_meta.items():
        pred[key] = value
    pred["prediction_source_file"] = relpath(pred_path)
    return pred


def run_one(
    job: Dict[str, Any],
    protocol: str,
    radius: float,
    particles: int,
    iterations: int,
    exp_config: Dict[str, Any],
    input_dir: Path,
    output_root: Path,
    objective: str,
) -> tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    """Run one synthetic NFPSO job."""

    run_seed = int(job["seed"])
    model = model_name_for_protocol(protocol)
    resolved_objective = objective_for_protocol(protocol, objective)

    csv_path = write_synthetic_csv(job, input_dir)
    run_name = (
        f"{job['dataset']}_{protocol}_obj{resolved_objective}_r{radius}_"
        f"p{particles}_i{iterations}_seed{run_seed}"
    ).replace(".", "p")
    run_dir = ensure_dir(output_root / "runs" / run_name)

    row_base: Dict[str, Any] = {
        "dataset": job["dataset"],
        "dataset_type": "synthetic",
        "base_name": job["base_name"],
        "generator": job["generator"],
        "length": job["length"],
        "noise_level": job["noise_level"],
        "seed": run_seed,
        "model": model,
        "source": "synthetic_nfpso_runner",
        "protocol": protocol,
        "objective": resolved_objective,
        "sc_radius": float(radius),
        "particles": int(particles),
        "iterations": int(iterations),
        "n_lags": int(job["n_lags"]),
        "season_length": int(job["season_length"]),
        "train_fraction": float(job["train_fraction"]),
        "synthetic_csv": relpath(csv_path),
        "run_dir": relpath(run_dir),
    }

    print(
        f"[RUN] dataset={job['dataset']} model={model} protocol={protocol} "
        f"radius={radius} seed={run_seed}"
    )

    start = time.perf_counter()
    status = "ok"
    error = None
    summary: Optional[Dict[str, Any]] = None

    try:
        cfg = make_experiment_config(
            csv_path=csv_path,
            output_dir=run_dir,
            protocol=protocol,
            objective=resolved_objective,
            radius=radius,
            seed=run_seed,
            particles=particles,
            iterations=iterations,
            n_lags=job["n_lags"],
            train_fraction=job["train_fraction"],
            exp_config=exp_config,
        )
        summary = run_experiment(cfg)
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"

    runtime = time.perf_counter() - start

    row: Dict[str, Any] = {
        **row_base,
        "runtime_seconds": runtime,
        "status": status,
        "error": error,
    }

    pred_df = None
    if summary is not None:
        row.update(copy_test_metrics(summary))
        row.update(flatten("initial_test", summary.get("initial_test_metrics")))
        row.update(flatten("train", summary.get("train_metrics")))
        row.update(flatten("validation", summary.get("validation_metrics")))
        row.update(flatten("test", summary.get("test_metrics")))
        row.update(flatten("diagnostics", summary.get("diagnostics")))
        row.update(flatten("initial_diagnostics", summary.get("initial_diagnostics")))
        row.update(summarize_history(run_dir / "pso_history.csv"))
        pred_df = read_predictions_if_available(run_dir, row_base)

    return row, pred_df


def merge_with_baselines(
    nfpso_metrics_path: Path,
    nfpso_predictions_path: Path,
    baseline_metrics_path: Path,
    baseline_predictions_path: Path,
    output_dir: Path,
) -> Dict[str, str]:
    """Merge synthetic baseline and synthetic NFPSO outputs."""

    ensure_dir(output_dir)

    metrics_frames: List[pd.DataFrame] = []
    prediction_frames: List[pd.DataFrame] = []

    if baseline_metrics_path.exists():
        base_metrics = pd.read_csv(baseline_metrics_path)
        if not base_metrics.empty:
            base_metrics["source"] = base_metrics.get("source", "synthetic_baseline_runner")
            metrics_frames.append(base_metrics)
    else:
        print(f"[SKIP] baseline metrics not found for merge: {baseline_metrics_path}")

    if nfpso_metrics_path.exists():
        nfpso_metrics = pd.read_csv(nfpso_metrics_path)
        if not nfpso_metrics.empty:
            metrics_frames.append(nfpso_metrics)

    if baseline_predictions_path.exists():
        base_pred = pd.read_csv(baseline_predictions_path)
        if not base_pred.empty:
            base_pred["source"] = base_pred.get("source", "synthetic_baseline_runner")
            prediction_frames.append(base_pred)

    if nfpso_predictions_path.exists():
        nfpso_pred = pd.read_csv(nfpso_predictions_path)
        if not nfpso_pred.empty:
            prediction_frames.append(nfpso_pred)

    metrics_path = output_dir / "metrics.csv"
    predictions_path = output_dir / "predictions.csv"

    if metrics_frames:
        pd.concat(metrics_frames, ignore_index=True, sort=False).to_csv(
            metrics_path,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        pd.DataFrame().to_csv(metrics_path, index=False, encoding="utf-8-sig")

    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True, sort=False).to_csv(
            predictions_path,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        pd.DataFrame().to_csv(predictions_path, index=False, encoding="utf-8-sig")

    return {
        "unified_metrics": str(metrics_path),
        "unified_predictions": str(predictions_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NFPSO on controlled synthetic time series.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Experiment config JSON.")
    parser.add_argument("--datasets-config", default=str(DEFAULT_DATASETS_CONFIG), help="Datasets config JSON.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--datasets", default=None, help="Comma-separated synthetic dataset names.")
    parser.add_argument("--lengths", default=None, help="Comma-separated synthetic lengths.")
    parser.add_argument("--noise-levels", default=None, help="Comma-separated noise levels.")
    parser.add_argument("--seeds", default=None, help="Comma-separated synthetic/NFPSO seeds.")
    parser.add_argument("--protocols", default=None, help="Comma-separated protocols: research_baseline,legacy_intent.")
    parser.add_argument("--objective", default="auto", choices=["auto", "error_std", "rmse", "composite"])
    parser.add_argument("--radii", default=None, help="Comma-separated SC radii.")
    parser.add_argument("--particles", type=int, default=None, help="Override particle count.")
    parser.add_argument("--iterations", type=int, default=None, help="Override iteration count.")
    parser.add_argument("--quick", action="store_true", help="Small smoke test.")
    parser.add_argument("--merge-with-baselines", action="store_true", help="Merge with existing synthetic baseline metrics.")
    parser.add_argument("--baseline-metrics", default=str(DEFAULT_SYNTHETIC_BASELINE_METRICS))
    parser.add_argument("--baseline-predictions", default=str(DEFAULT_SYNTHETIC_BASELINE_PREDICTIONS))
    args = parser.parse_args()

    exp_config = load_json(Path(args.config))
    datasets_config = load_json(Path(args.datasets_config))

    output_base = Path(args.output_dir or exp_config.get("output_dir", "outputs/q1_minimal"))
    output_root = ensure_dir(ROOT / output_base / "synthetic_nfpso_experiments")
    input_dir = ensure_dir(output_root / "synthetic_inputs")

    jobs = resolve_synthetic_jobs(
        exp_config=exp_config,
        datasets_config=datasets_config,
        cli_datasets=parse_csv_list(args.datasets),
        cli_lengths=parse_int_list(args.lengths),
        cli_noise_levels=parse_float_list(args.noise_levels),
        cli_seeds=parse_int_list(args.seeds),
        quick=args.quick,
    )
    protocols = resolve_protocols(parse_csv_list(args.protocols), quick=args.quick)
    radii = resolve_radii(exp_config, parse_float_list(args.radii), quick=args.quick)
    particles, iterations = resolve_particles_iterations(
        exp_config,
        cli_particles=args.particles,
        cli_iterations=args.iterations,
        quick=args.quick,
    )

    if not jobs:
        print("[ERROR] No synthetic jobs were created.")
        return 2

    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] synthetic_jobs={len(jobs)} protocols={protocols} radii={radii}")
    print(f"[INFO] particles={particles} iterations={iterations}")

    metric_rows: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []

    for job in jobs:
        for protocol in protocols:
            for radius in radii:
                row, pred = run_one(
                    job=job,
                    protocol=protocol,
                    radius=float(radius),
                    particles=particles,
                    iterations=iterations,
                    exp_config=exp_config,
                    input_dir=input_dir,
                    output_root=output_root,
                    objective=args.objective,
                )
                metric_rows.append(row)
                if pred is not None:
                    prediction_frames.append(pred)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = output_root / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    predictions_path = output_root / "predictions.csv"
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True, sort=False).to_csv(
            predictions_path,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        pd.DataFrame().to_csv(predictions_path, index=False, encoding="utf-8-sig")

    manifest_path = output_root / "manifest.csv"
    manifest_cols = [
        c for c in [
            "dataset",
            "model",
            "base_name",
            "generator",
            "length",
            "noise_level",
            "seed",
            "protocol",
            "objective",
            "sc_radius",
            "particles",
            "iterations",
            "runtime_seconds",
            "status",
            "error",
            "synthetic_csv",
            "run_dir",
        ]
        if c in metrics_df.columns
    ]
    metrics_df[manifest_cols].to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] metrics: {metrics_path}")
    print(f"[DONE] predictions: {predictions_path}")
    print(f"[DONE] manifest: {manifest_path}")

    if args.merge_with_baselines:
        unified_dir = ensure_dir(ROOT / output_base / "synthetic_unified_benchmarks")
        paths = merge_with_baselines(
            nfpso_metrics_path=metrics_path,
            nfpso_predictions_path=predictions_path,
            baseline_metrics_path=Path(args.baseline_metrics),
            baseline_predictions_path=Path(args.baseline_predictions),
            output_dir=unified_dir,
        )
        print(f"[DONE] unified metrics: {paths['unified_metrics']}")
        print(f"[DONE] unified predictions: {paths['unified_predictions']}")

    if "status" in metrics_df.columns:
        failed = metrics_df[metrics_df["status"].fillna("ok").astype(str).str.lower().ne("ok")]
        if not failed.empty:
            print(f"[WARN] failed rows: {len(failed)}")
            show_cols = [c for c in ["dataset", "model", "protocol", "error"] if c in failed.columns]
            print(failed[show_cols].to_string(index=False))

    print("[NOTE] outputs/ is ignored by Git and should not be committed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
