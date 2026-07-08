"""
Run NFPSO experiments for the SS-NFPSO Q1 pipeline.

This script wraps the existing `nfpsosc.experiment.run_experiment` function and
exports a statistical-testing-friendly metrics table.

Main outputs
------------
    outputs/q1_minimal/nfpso_experiments/metrics.csv
    outputs/q1_minimal/nfpso_experiments/predictions.csv
    outputs/q1_minimal/nfpso_experiments/manifest.csv

Design choices
--------------
- The default local CSV is `data/tax_revenues.csv`, because this is the user's
  private local data file. It must remain ignored by Git.
- Each NFPSO run gets its own subdirectory under outputs/.
- The output `metrics.csv` uses the same long-table style as the baseline
  runners: dataset, model, rmse, mae, smape_percent, mase, ...
- Real publication claims should be based on full runs, not quick smoke tests.

Examples
--------
Quick smoke test:

    py run_nfpso_experiments.py --quick

Research baseline only:

    py run_nfpso_experiments.py --protocols research_baseline

Legacy and research comparison:

    py run_nfpso_experiments.py --protocols legacy_intent,research_baseline

Custom seeds and radii:

    py run_nfpso_experiments.py --seeds 1,2,3 --radii 0.35,0.55,0.75

After this script, rerun:

    py run_statistical_tests.py --inputs outputs/q1_minimal/nfpso_experiments/metrics.csv --reference-model projected_constricted_nfpso
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nfpsosc.experiment import ExperimentConfig, run_experiment


DEFAULT_CONFIG = ROOT / "configs" / "q1_minimal_experiments.json"
DEFAULT_CSV = ROOT / "data" / "tax_revenues.csv"
FALLBACK_CSV = ROOT / "data" / "yemen_tax_revenues_2002_2014.csv"


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON config if it exists; otherwise return an empty dictionary."""

    if not path.exists():
        print(f"[WARN] config not found, using defaults: {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    """Create and return directory."""

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
    """Return path relative to project root when possible."""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_csv(cli_csv: Optional[str]) -> Path:
    """Resolve local data CSV."""

    if cli_csv:
        return Path(cli_csv)

    if DEFAULT_CSV.exists():
        return DEFAULT_CSV
    return FALLBACK_CSV


def resolve_protocols(cli_protocols: Optional[List[str]], quick: bool) -> List[str]:
    """Resolve protocol list."""

    if cli_protocols:
        protocols = cli_protocols
    else:
        protocols = ["research_baseline", "legacy_intent"]

    valid = {"legacy_intent", "research_baseline"}
    invalid = [p for p in protocols if p not in valid]
    if invalid:
        raise ValueError(f"Invalid protocols: {invalid}. Valid protocols: {sorted(valid)}")

    if quick:
        # Keep the smoke test short and focused on the research version.
        if "research_baseline" in protocols:
            return ["research_baseline"]
        return protocols[:1]

    return protocols


def resolve_seeds(config: Dict[str, Any], cli_seeds: Optional[List[int]], quick: bool) -> List[int]:
    """Resolve seeds from CLI/config/defaults."""

    if cli_seeds is not None:
        seeds = cli_seeds
    else:
        seeds = list(config.get("seed_study", {}).get("seeds", config.get("random_seeds", [1, 2, 3])))

    seeds = [int(s) for s in seeds]
    if quick:
        return seeds[:1] or [1]
    return seeds


def resolve_radii(config: Dict[str, Any], cli_radii: Optional[List[float]], quick: bool) -> List[float]:
    """Resolve SC radii from CLI/config/defaults."""

    if cli_radii is not None:
        radii = cli_radii
    else:
        radii = (
            config.get("radius_sweep", {}).get("radii")
            or config.get("stability_sweep", {}).get("radii")
            or config.get("nfpso", {}).get("radii")
            or [0.35, 0.55, 0.75]
        )

    radii = [float(r) for r in radii]
    if quick:
        # Use the middle/default radius when possible.
        return [0.55 if 0.55 in radii else radii[0]]
    return radii


def resolve_particles_iterations(
    config: Dict[str, Any],
    cli_particles: Optional[int],
    cli_iterations: Optional[int],
    quick: bool,
) -> tuple[int, int]:
    """Resolve NFPSO particle and iteration budget."""

    nfpso_cfg = config.get("nfpso", {})
    particles = int(cli_particles if cli_particles is not None else nfpso_cfg.get("particles", 12))
    iterations = int(cli_iterations if cli_iterations is not None else nfpso_cfg.get("iterations", 40))

    if quick:
        particles = min(particles, 5)
        iterations = min(iterations, 5)

    return particles, iterations


def objective_for_protocol(protocol: str, objective: str) -> str:
    """Resolve objective name."""

    if objective != "auto":
        return objective

    if protocol == "legacy_intent":
        return "error_std"
    return "composite"


def model_name_for_protocol(protocol: str) -> str:
    """Map experiment protocol to model name for metrics tables."""

    if protocol == "legacy_intent":
        return "legacy_nfpso"
    if protocol == "research_baseline":
        return "projected_constricted_nfpso"
    return protocol


def flatten(prefix: str, values: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten dictionary with a prefix."""

    if not values:
        return {}

    out: Dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[f"{prefix}_{key}"] = value
    return out


def copy_test_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Copy test metrics to unprefixed columns expected by statistical scripts.

    Example:
        summary['test_metrics']['rmse'] -> row['rmse']
    """

    metrics = summary.get("test_metrics") or {}
    out: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) or value is None:
            out[key] = value
    return out


def summarize_history(history_path: Path) -> Dict[str, Any]:
    """Summarize optimizer history if pso_history.csv exists."""

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
    """Read test predictions from a run directory if present."""

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


def make_experiment_config(
    csv_path: Path,
    output_dir: Path,
    protocol: str,
    objective: str,
    radius: float,
    seed: int,
    particles: int,
    iterations: int,
    config: Dict[str, Any],
) -> ExperimentConfig:
    """Create ExperimentConfig using known repository fields."""

    nfpso_cfg = config.get("nfpso", {})

    return ExperimentConfig(
        csv_path=str(csv_path),
        output_dir=str(output_dir),
        n_lags=int(nfpso_cfg.get("default_n_lags", 5)),
        train_fraction=float(config.get("train_fraction", 0.85)),
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


def run_one_experiment(
    csv_path: Path,
    output_root: Path,
    protocol: str,
    objective: str,
    radius: float,
    seed: int,
    particles: int,
    iterations: int,
    config: Dict[str, Any],
) -> tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    """Run one NFPSO experiment and return metrics row plus optional predictions."""

    model = model_name_for_protocol(protocol)
    dataset = f"yemen_tax_revenue_{protocol}_r{radius}_seed{seed}"
    run_name = f"{protocol}_obj{objective}_r{radius}_p{particles}_i{iterations}_seed{seed}"
    run_name = run_name.replace(".", "p")
    run_dir = ensure_dir(output_root / "runs" / run_name)

    row_base: Dict[str, Any] = {
        "dataset": dataset,
        "dataset_type": "real",
        "model": model,
        "protocol": protocol,
        "objective": objective,
        "sc_radius": float(radius),
        "seed": int(seed),
        "particles": int(particles),
        "iterations": int(iterations),
        "csv_path": relpath(csv_path),
        "run_dir": relpath(run_dir),
    }

    print(
        f"[RUN] model={model} protocol={protocol} objective={objective} "
        f"radius={radius} seed={seed} particles={particles} iterations={iterations}"
    )

    start_time = time.perf_counter()
    summary: Optional[Dict[str, Any]] = None
    status = "ok"
    error_message = None

    try:
        exp_cfg = make_experiment_config(
            csv_path=csv_path,
            output_dir=run_dir,
            protocol=protocol,
            objective=objective,
            radius=radius,
            seed=seed,
            particles=particles,
            iterations=iterations,
            config=config,
        )
        summary = run_experiment(exp_cfg)
    except Exception as exc:
        status = "failed"
        error_message = f"{type(exc).__name__}: {exc}"

    runtime = time.perf_counter() - start_time

    row: Dict[str, Any] = {
        **row_base,
        "runtime_seconds": runtime,
        "status": status,
        "error": error_message,
    }

    prediction_frame = None

    if summary is not None:
        row.update(copy_test_metrics(summary))
        row.update(flatten("initial_test", summary.get("initial_test_metrics")))
        row.update(flatten("train", summary.get("train_metrics")))
        row.update(flatten("validation", summary.get("validation_metrics")))
        row.update(flatten("test", summary.get("test_metrics")))
        row.update(flatten("diagnostics", summary.get("diagnostics")))
        row.update(flatten("initial_diagnostics", summary.get("initial_diagnostics")))
        row.update(summarize_history(run_dir / "pso_history.csv"))
        prediction_frame = read_predictions_if_available(run_dir, row_base)

    return row, prediction_frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NFPSO experiments and export comparable metrics.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Experiment config JSON.")
    parser.add_argument("--csv", default=None, help="Local private data CSV. Default: data/tax_revenues.csv.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--protocols", default=None, help="Comma-separated protocols: legacy_intent,research_baseline.")
    parser.add_argument("--objective", default="auto", choices=["auto", "error_std", "rmse", "composite"])
    parser.add_argument("--radii", default=None, help="Comma-separated SC radii, e.g. 0.35,0.55,0.75.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds, e.g. 1,2,3.")
    parser.add_argument("--particles", type=int, default=None, help="Override particle count.")
    parser.add_argument("--iterations", type=int, default=None, help="Override iteration count.")
    parser.add_argument("--quick", action="store_true", help="Small smoke test: first seed, one radius, tiny budget.")
    args = parser.parse_args()

    config = load_json(Path(args.config))
    csv_path = resolve_csv(args.csv)

    if not csv_path.exists():
        print(f"[SKIP] CSV not found: {csv_path}")
        print("[NOTE] Use --csv data\\tax_revenues.csv or place the private CSV locally.")
        return 0

    protocols = resolve_protocols(parse_csv_list(args.protocols), quick=args.quick)
    radii = resolve_radii(config, parse_float_list(args.radii), quick=args.quick)
    seeds = resolve_seeds(config, parse_int_list(args.seeds), quick=args.quick)
    particles, iterations = resolve_particles_iterations(
        config,
        cli_particles=args.particles,
        cli_iterations=args.iterations,
        quick=args.quick,
    )

    output_root = Path(args.output_dir or config.get("output_dir", "outputs/q1_minimal"))
    output_root = ensure_dir(ROOT / output_root / "nfpso_experiments")

    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] csv={csv_path}")
    print(f"[INFO] protocols={protocols}")
    print(f"[INFO] radii={radii}")
    print(f"[INFO] seeds={seeds}")
    print(f"[INFO] particles={particles} iterations={iterations}")

    metric_rows: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []

    for protocol in protocols:
        objective = objective_for_protocol(protocol, args.objective)
        for radius in radii:
            for seed in seeds:
                row, pred = run_one_experiment(
                    csv_path=csv_path,
                    output_root=output_root,
                    protocol=protocol,
                    objective=objective,
                    radius=float(radius),
                    seed=int(seed),
                    particles=particles,
                    iterations=iterations,
                    config=config,
                )
                metric_rows.append(row)
                if pred is not None:
                    prediction_frames.append(pred)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = output_root / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    predictions_path = output_root / "predictions.csv"
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
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
            "protocol",
            "objective",
            "sc_radius",
            "seed",
            "particles",
            "iterations",
            "runtime_seconds",
            "status",
            "error",
            "run_dir",
        ]
        if c in metrics_df.columns
    ]
    metrics_df[manifest_cols].to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] metrics: {metrics_path}")
    print(f"[DONE] predictions: {predictions_path}")
    print(f"[DONE] manifest: {manifest_path}")

    failed = metrics_df[metrics_df["status"] != "ok"] if "status" in metrics_df.columns else pd.DataFrame()
    if not failed.empty:
        print(f"[WARN] failed runs: {len(failed)}")
        print(failed[["dataset", "model", "error"]].to_string(index=False))

    print("[NOTE] outputs/ is ignored by Git and should not be committed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
