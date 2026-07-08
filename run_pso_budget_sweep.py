"""
Run PSO-budget sensitivity experiments for the SS-NFPSO Q1 pipeline.

This script repeatedly calls the existing NFPSO experiment runner with different
particle/iteration budgets and collects comparable metrics and diagnostics.

Important
---------
- This script currently targets the local authorized Yemen tax-revenue series,
  because the current `nfpsosc.experiment.run_experiment` implementation expects
  the revenue CSV schema.
- If the private CSV is missing, the script exits safely without creating claims.
- Outputs are written under outputs/, which should remain ignored by Git.

Examples
--------
Quick smoke test:

    py run_pso_budget_sweep.py --quick

Configured minimal run:

    py run_pso_budget_sweep.py --config configs/q1_minimal_experiments.json

Custom budgets:

    py run_pso_budget_sweep.py --budgets 5x5,10x20,12x40 --seeds 1,2,3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nfpsosc.experiment import ExperimentConfig, run_experiment


DEFAULT_CONFIG = ROOT / "configs" / "q1_minimal_experiments.json"
DEFAULT_CSV = ROOT / "data" / "yemen_tax_revenues_2002_2014.csv"


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON config file."""

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    """Create a directory and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated integer list."""

    if value is None or value.strip() == "":
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_budgets(value: Optional[str]) -> Optional[List[Dict[str, int]]]:
    """
    Parse budgets such as '5x5,10x20,12x40'.

    Returns a list of dictionaries with keys: particles, iterations.
    """

    if value is None or value.strip() == "":
        return None

    budgets: List[Dict[str, int]] = []
    for token in value.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if "x" not in token:
            raise ValueError(f"Budget '{token}' must have the form particlesxiterations")
        particles, iterations = token.split("x", 1)
        budgets.append({"particles": int(particles), "iterations": int(iterations)})
    return budgets


def flatten_metrics(prefix: str, metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten a metric dictionary with a prefix."""

    if not metrics:
        return {}
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def flatten_diagnostics(prefix: str, diagnostics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten diagnostics with a prefix."""

    if not diagnostics:
        return {}
    return {f"{prefix}_{key}": value for key, value in diagnostics.items()}


def resolve_budgets(config: Dict[str, Any], cli_budgets: Optional[List[Dict[str, int]]], quick: bool) -> List[Dict[str, int]]:
    """Resolve budget list from CLI or config."""

    if cli_budgets is not None:
        budgets = cli_budgets
    else:
        budgets = list(config.get("pso_budget_sweep", {}).get("budgets", []))
        if not budgets:
            budgets = [
                {"particles": 5, "iterations": 5},
                {"particles": 10, "iterations": 20},
                {"particles": 12, "iterations": 40},
            ]

    normalized = [
        {"particles": int(b["particles"]), "iterations": int(b["iterations"])}
        for b in budgets
    ]

    if quick:
        return normalized[:2]

    return normalized


def resolve_seeds(config: Dict[str, Any], cli_seeds: Optional[List[int]], quick: bool) -> List[int]:
    """Resolve seeds from CLI or config."""

    if cli_seeds is not None:
        seeds = cli_seeds
    else:
        seeds = list(config.get("seed_study", {}).get("seeds", config.get("random_seeds", [123])))

    seeds = [int(s) for s in seeds]
    if quick:
        return seeds[:1] or [1]
    return seeds


def summarize_history(history_path: Path) -> Dict[str, Any]:
    """
    Summarize PSO history CSV written by run_experiment, if available.

    The exact column names depend on the optimizer implementation, so this
    function extracts only columns that exist.
    """

    if not history_path.exists():
        return {}

    try:
        hist = pd.read_csv(history_path)
    except Exception:
        return {}

    out: Dict[str, Any] = {"history_rows": int(len(hist))}

    possible_last_cols = [
        "best_cost",
        "mean_velocity_norm",
        "swarm_diameter",
        "best_parameter_drift",
        "boundary_hits",
    ]
    for col in possible_last_cols:
        if col in hist.columns and len(hist):
            out[f"history_final_{col}"] = float(hist[col].iloc[-1])
            out[f"history_max_{col}"] = float(hist[col].max())
            out[f"history_min_{col}"] = float(hist[col].min())

    if "best_cost" in hist.columns and len(hist):
        out["history_initial_best_cost"] = float(hist["best_cost"].iloc[0])
        out["history_final_best_cost"] = float(hist["best_cost"].iloc[-1])
        out["history_best_cost_improvement"] = float(hist["best_cost"].iloc[0] - hist["best_cost"].iloc[-1])

    return out


def make_run_config(
    csv_path: Path,
    output_dir: Path,
    protocol: str,
    objective: str,
    particles: int,
    iterations: int,
    seed: int,
    radius: float,
    exp_config: Dict[str, Any],
) -> ExperimentConfig:
    """Create ExperimentConfig for one budget run."""

    nfpso_cfg = exp_config.get("nfpso", {})

    return ExperimentConfig(
        csv_path=str(csv_path),
        output_dir=str(output_dir),
        n_lags=int(nfpso_cfg.get("default_n_lags", 5)),
        train_fraction=float(exp_config.get("train_fraction", 0.85)) if "train_fraction" in exp_config else 0.85,
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


def run_budget_sweep(
    csv_path: Path,
    output_root: Path,
    budgets: List[Dict[str, int]],
    seeds: List[int],
    protocol: str,
    objective: str,
    radius: float,
    exp_config: Dict[str, Any],
) -> pd.DataFrame:
    """Run the full budget sweep and return a results DataFrame."""

    rows: List[Dict[str, Any]] = []

    for budget in budgets:
        particles = int(budget["particles"])
        iterations = int(budget["iterations"])

        for seed in seeds:
            run_name = f"{protocol}_p{particles}_i{iterations}_seed{seed}"
            run_dir = ensure_dir(output_root / "runs" / run_name)
            print(f"[RUN] particles={particles} iterations={iterations} seed={seed}")

            start = time.perf_counter()
            status = "ok"
            error_message = None
            summary: Optional[Dict[str, Any]] = None

            try:
                config = make_run_config(
                    csv_path=csv_path,
                    output_dir=run_dir,
                    protocol=protocol,
                    objective=objective,
                    particles=particles,
                    iterations=iterations,
                    seed=seed,
                    radius=radius,
                    exp_config=exp_config,
                )
                summary = run_experiment(config)
            except Exception as exc:
                status = "failed"
                error_message = f"{type(exc).__name__}: {exc}"

            runtime = time.perf_counter() - start

            row: Dict[str, Any] = {
                "run_name": run_name,
                "protocol": protocol,
                "objective": objective,
                "particles": particles,
                "iterations": iterations,
                "seed": seed,
                "sc_radius": radius,
                "runtime_seconds": runtime,
                "status": status,
                "error": error_message,
                "output_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
            }

            if summary is not None:
                row.update(flatten_metrics("initial_test", summary.get("initial_test_metrics")))
                row.update(flatten_metrics("train", summary.get("train_metrics")))
                row.update(flatten_metrics("validation", summary.get("validation_metrics")))
                row.update(flatten_metrics("test", summary.get("test_metrics")))
                row.update(flatten_diagnostics("diagnostics", summary.get("diagnostics")))
                row.update(flatten_diagnostics("initial_diagnostics", summary.get("initial_diagnostics")))
                row.update(summarize_history(run_dir / "pso_history.csv"))

            rows.append(row)

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PSO budget sensitivity experiments.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Experiment config JSON.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Local authorized revenue CSV.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--protocol", choices=["legacy_intent", "research_baseline"], default="research_baseline")
    parser.add_argument("--objective", choices=["error_std", "rmse", "composite"], default=None)
    parser.add_argument("--radius", type=float, default=0.55)
    parser.add_argument("--budgets", default=None, help="Comma-separated budgets such as 5x5,10x20,12x40.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds such as 1,2,3.")
    parser.add_argument("--quick", action="store_true", help="Run only the first two budgets and first seed.")
    args = parser.parse_args()

    config_path = Path(args.config)
    exp_config = load_json(config_path)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[SKIP] CSV not found: {csv_path}")
        print("[NOTE] Place the authorized local data file at data/yemen_tax_revenues_2002_2014.csv to run this sweep.")
        return 0

    objective = args.objective
    if objective is None:
        objective = "error_std" if args.protocol == "legacy_intent" else "composite"

    budgets = resolve_budgets(exp_config, parse_budgets(args.budgets), quick=args.quick)
    seeds = resolve_seeds(exp_config, parse_int_list(args.seeds), quick=args.quick)

    output_root = Path(args.output_dir or exp_config.get("output_dir", "outputs/q1_minimal"))
    output_root = ensure_dir(ROOT / output_root / "pso_budget")

    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] protocol={args.protocol} objective={objective} radius={args.radius}")
    print(f"[INFO] budgets={budgets}")
    print(f"[INFO] seeds={seeds}")

    results = run_budget_sweep(
        csv_path=csv_path,
        output_root=output_root,
        budgets=budgets,
        seeds=seeds,
        protocol=args.protocol,
        objective=objective,
        radius=float(args.radius),
        exp_config=exp_config,
    )

    results_path = output_root / "pso_budget_results.csv"
    results.to_csv(results_path, index=False, encoding="utf-8-sig")

    # Compact summary by budget for paper planning.
    summary_path = output_root / "pso_budget_summary.csv"
    if not results.empty and "test_rmse" in results.columns:
        ok = results[results["status"] == "ok"].copy()
        group_cols = ["protocol", "objective", "particles", "iterations", "sc_radius"]
        value_cols = [
            c for c in [
                "test_rmse",
                "test_mae",
                "test_mape_percent",
                "test_smape_percent",
                "test_mase",
                "diagnostics_jacobian_norm_max",
                "diagnostics_empirical_lipschitz_max",
                "diagnostics_parameter_l2_norm",
                "runtime_seconds",
            ]
            if c in ok.columns
        ]
        if value_cols:
            summary_df = ok.groupby(group_cols, as_index=False)[value_cols].agg(["mean", "std"])
            summary_df.columns = ["_".join([str(x) for x in col if x]) for col in summary_df.columns.to_flat_index()]
            summary_df = summary_df.reset_index()
            summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame().to_csv(summary_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] results: {results_path}")
    print(f"[DONE] summary: {summary_path}")
    print("[NOTE] outputs/ is ignored by Git and should not be committed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
