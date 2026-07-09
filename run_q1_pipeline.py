"""
Run the SS-NFPSO Q1 reproducibility pipeline.

This is an orchestration script. It calls the existing pipeline scripts in a
safe order and writes a final manifest.

Stages
------
1. baselines              -> run_baselines.py
2. synthetic_benchmarks   -> run_synthetic_benchmarks.py
3. nfpso_experiments      -> run_nfpso_experiments.py
4. real_data_benchmarks   -> run_real_data_benchmarks.py
5. pso_budget             -> run_pso_budget_sweep.py
6. statistical_tests      -> run_statistical_tests.py

Default behavior
----------------
- Uses the local private file: data/tax_revenues.csv
- Writes all generated outputs under outputs/
- Does not commit or upload outputs
- Continues or stops depending on --continue-on-error
- In --quick mode, keeps all experiments small for smoke testing

Examples
--------
Quick smoke test:

    py run_q1_pipeline.py --quick --csv data\\tax_revenues.csv

Run selected stages only:

    py run_q1_pipeline.py --quick --stages baselines,synthetic_benchmarks,statistics

Run without PSO budget sweep:

    py run_q1_pipeline.py --quick --skip-stages pso_budget

Full local run:

    py run_q1_pipeline.py --csv data\\tax_revenues.csv

Then check Git:

    git status

Only this script should be committed. outputs/ and private CSV files must remain
ignored by Git.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parent

DEFAULT_CONFIG = ROOT / "configs" / "q1_minimal_experiments.json"
DEFAULT_CSV = ROOT / "data" / "tax_revenues.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "q1_minimal"

ALL_STAGES = [
    "baselines",
    "synthetic_benchmarks",
    "synthetic_nfpso_experiments",
    "nfpso_experiments",
    "real_data_benchmarks",
    "pso_budget",
    "statistical_tests",
]


@dataclass
class StageResult:
    """Serializable stage execution result."""

    stage: str
    command: str
    status: str
    return_code: int
    runtime_seconds: float
    started_at_epoch: float
    finished_at_epoch: float
    error: Optional[str] = None


def parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated CLI values."""

    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def ensure_dir(path: Path) -> Path:
    """Create and return directory."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def relpath(path: Path) -> str:
    """Return a path relative to the project root when possible."""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_stages(stages: Optional[List[str]], skip_stages: Optional[List[str]]) -> List[str]:
    """Resolve stage list."""

    selected = stages or list(ALL_STAGES)
    selected = [s.strip() for s in selected if s.strip()]

    invalid = [s for s in selected if s not in ALL_STAGES]
    if invalid:
        raise ValueError(f"Invalid stages: {invalid}. Valid stages: {ALL_STAGES}")

    skips = set(skip_stages or [])
    invalid_skips = [s for s in skips if s not in ALL_STAGES]
    if invalid_skips:
        raise ValueError(f"Invalid skip stages: {invalid_skips}. Valid stages: {ALL_STAGES}")

    return [s for s in selected if s not in skips]


def python_command(script_name: str, *args: str) -> List[str]:
    """Build a command using the active Python interpreter."""

    return [sys.executable, str(ROOT / script_name), *args]


def command_to_text(command: Sequence[str]) -> str:
    """Return a display-friendly command."""

    return " ".join(f'"{part}"' if " " in part else part for part in command)


def run_command(
    stage: str,
    command: Sequence[str],
    continue_on_error: bool,
    env: Optional[Dict[str, str]] = None,
) -> StageResult:
    """Run a stage command and return a StageResult."""

    command_list = list(command)
    command_text = command_to_text(command_list)
    print(f"\n[STAGE] {stage}")
    print(f"[CMD] {command_text}")

    started = time.time()
    start_perf = time.perf_counter()
    status = "ok"
    error = None
    return_code = 0

    try:
        completed = subprocess.run(
            command_list,
            cwd=str(ROOT),
            env=env,
            text=True,
            check=False,
        )
        return_code = int(completed.returncode)
        if return_code != 0:
            status = "failed"
            error = f"return_code={return_code}"
    except Exception as exc:
        status = "failed"
        return_code = -1
        error = f"{type(exc).__name__}: {exc}"

    finished = time.time()
    runtime = time.perf_counter() - start_perf

    result = StageResult(
        stage=stage,
        command=command_text,
        status=status,
        return_code=return_code,
        runtime_seconds=runtime,
        started_at_epoch=started,
        finished_at_epoch=finished,
        error=error,
    )

    if status != "ok":
        print(f"[FAILED] {stage}: {error}")
        if not continue_on_error:
            raise RuntimeError(f"Stage failed: {stage}: {error}")
    else:
        print(f"[OK] {stage} runtime_seconds={runtime:.2f}")

    return result


def make_env() -> Dict[str, str]:
    """Return environment with PYTHONPATH=src."""

    env = os.environ.copy()
    src = str(ROOT / "src")
    old = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not old else src + os.pathsep + old
    return env


def stage_command(
    stage: str,
    args: argparse.Namespace,
    output_root: Path,
) -> List[str]:
    """Create the command for a pipeline stage."""

    config = str(args.config)
    csv_path = str(args.csv)
    quick_flag = ["--quick"] if args.quick else []

    if stage == "baselines":
        cmd = python_command(
            "run_baselines.py",
            "--config",
            config,
            "--output-dir",
            relpath(output_root),
            *quick_flag,
        )
        if args.baseline_models:
            cmd.extend(["--models", args.baseline_models])
        return cmd

    if stage == "synthetic_benchmarks":
        cmd = python_command(
            "run_synthetic_benchmarks.py",
            "--config",
            config,
            "--output-dir",
            relpath(output_root),
            *quick_flag,
        )
        if args.synthetic_models:
            cmd.extend(["--models", args.synthetic_models])
        elif args.baseline_models:
            cmd.extend(["--models", args.baseline_models])
        return cmd

    if stage == "synthetic_nfpso_experiments":
        cmd = python_command(
            "run_synthetic_nfpso_experiments.py",
            "--config",
            config,
            "--output-dir",
            relpath(output_root),
            "--merge-with-baselines",
            *quick_flag,
        )
        if args.nfpso_protocols:
            cmd.extend(["--protocols", args.nfpso_protocols])
        if args.nfpso_radii:
            cmd.extend(["--radii", args.nfpso_radii])
        if args.nfpso_seeds:
            cmd.extend(["--seeds", args.nfpso_seeds])
        if args.nfpso_particles is not None:
            cmd.extend(["--particles", str(args.nfpso_particles)])
        if args.nfpso_iterations is not None:
            cmd.extend(["--iterations", str(args.nfpso_iterations)])
        return cmd

    if stage == "nfpso_experiments":
        cmd = python_command(
            "run_nfpso_experiments.py",
            "--config",
            config,
            "--csv",
            csv_path,
            "--output-dir",
            relpath(output_root),
            *quick_flag,
        )
        if args.nfpso_protocols:
            cmd.extend(["--protocols", args.nfpso_protocols])
        if args.nfpso_radii:
            cmd.extend(["--radii", args.nfpso_radii])
        if args.nfpso_seeds:
            cmd.extend(["--seeds", args.nfpso_seeds])
        if args.nfpso_particles is not None:
            cmd.extend(["--particles", str(args.nfpso_particles)])
        if args.nfpso_iterations is not None:
            cmd.extend(["--iterations", str(args.nfpso_iterations)])
        return cmd

    if stage == "real_data_benchmarks":
        cmd = python_command(
            "run_real_data_benchmarks.py",
            "--config",
            config,
            "--csv",
            csv_path,
            "--output-dir",
            relpath(output_root),
            "--dataset-name",
            args.dataset_name,
            *quick_flag,
        )
        if args.real_models:
            cmd.extend(["--models", args.real_models])
        elif args.baseline_models:
            cmd.extend(["--models", args.baseline_models])
        if args.nfpso_source:
            cmd.extend(["--nfpso-source", args.nfpso_source])
        if args.nfpso_protocols:
            cmd.extend(["--protocols", args.nfpso_protocols])
        if args.nfpso_radii:
            cmd.extend(["--radii", args.nfpso_radii])
        if args.nfpso_seeds:
            cmd.extend(["--seeds", args.nfpso_seeds])
        return cmd

    if stage == "pso_budget":
        cmd = python_command(
            "run_pso_budget_sweep.py",
            "--config",
            config,
            "--csv",
            csv_path,
            "--output-dir",
            relpath(output_root),
            *quick_flag,
        )
        if args.pso_budgets:
            cmd.extend(["--budgets", args.pso_budgets])
        if args.pso_seeds:
            cmd.extend(["--seeds", args.pso_seeds])
        return cmd

    if stage == "statistical_tests":
        # Use the unified real-data benchmark and synthetic benchmark tables.
        default_inputs = [
            output_root / "real_data_benchmarks" / "metrics.csv",
            output_root / "synthetic_unified_benchmarks" / "metrics.csv",
        ]
        inputs = args.statistics_inputs or ",".join(relpath(p) for p in default_inputs)

        cmd = python_command(
            "run_statistical_tests.py",
            "--inputs",
            inputs,
            "--output-dir",
            relpath(output_root / "statistical_tests"),
            "--reference-model",
            args.reference_model,
            *quick_flag,
        )
        if args.statistics_metrics:
            cmd.extend(["--metrics", args.statistics_metrics])
        return cmd

    raise ValueError(f"Unsupported stage: {stage}")


def write_manifest(results: List[StageResult], output_dir: Path, payload: Dict[str, Any]) -> Dict[str, str]:
    """Write JSON and CSV pipeline manifest files."""

    ensure_dir(output_dir)

    results_dicts = [asdict(r) for r in results]

    manifest_json = output_dir / "q1_pipeline_manifest.json"
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                **payload,
                "stage_results": results_dicts,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    manifest_csv = output_dir / "q1_pipeline_manifest.csv"
    with manifest_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "stage",
            "command",
            "status",
            "return_code",
            "runtime_seconds",
            "started_at_epoch",
            "finished_at_epoch",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_dicts:
            writer.writerow(row)

    return {
        "manifest_json": str(manifest_json),
        "manifest_csv": str(manifest_csv),
    }


def validate_script_files(stages: Sequence[str]) -> List[str]:
    """Return missing scripts required by selected stages."""

    scripts_by_stage = {
        "baselines": "run_baselines.py",
        "synthetic_benchmarks": "run_synthetic_benchmarks.py",
        "synthetic_nfpso_experiments": "run_synthetic_nfpso_experiments.py",
        "nfpso_experiments": "run_nfpso_experiments.py",
        "real_data_benchmarks": "run_real_data_benchmarks.py",
        "pso_budget": "run_pso_budget_sweep.py",
        "statistical_tests": "run_statistical_tests.py",
    }

    missing = []
    for stage in stages:
        script = ROOT / scripts_by_stage[stage]
        if not script.exists():
            missing.append(str(script))
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SS-NFPSO Q1 reproducibility pipeline.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Experiment config JSON.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Local private real-data CSV.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Pipeline output root.")
    parser.add_argument("--dataset-name", default="yemen_tax_revenue", help="Unified real-data dataset name.")
    parser.add_argument("--stages", default=None, help=f"Comma-separated stages. Valid: {','.join(ALL_STAGES)}")
    parser.add_argument("--skip-stages", default=None, help="Comma-separated stages to skip.")
    parser.add_argument("--quick", action="store_true", help="Run smoke-test-sized experiments.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after a failed stage.")

    parser.add_argument("--baseline-models", default=None, help="Models for run_baselines.py and synthetic unless overridden.")
    parser.add_argument("--synthetic-models", default=None, help="Models for synthetic benchmarks.")
    parser.add_argument("--real-models", default=None, help="Baseline models for real-data benchmarks.")

    parser.add_argument("--nfpso-source", default=None, choices=["auto", "existing", "run", "skip"], help="NFPSO source in real-data benchmark.")
    parser.add_argument("--nfpso-protocols", default=None, help="Comma-separated NFPSO protocols.")
    parser.add_argument("--nfpso-radii", default=None, help="Comma-separated NFPSO radii.")
    parser.add_argument("--nfpso-seeds", default=None, help="Comma-separated NFPSO seeds.")
    parser.add_argument("--nfpso-particles", type=int, default=None, help="NFPSO particle count.")
    parser.add_argument("--nfpso-iterations", type=int, default=None, help="NFPSO iteration count.")

    parser.add_argument("--pso-budgets", default=None, help="PSO budgets such as 5x5,10x20.")
    parser.add_argument("--pso-seeds", default=None, help="PSO budget sweep seeds.")
    parser.add_argument("--statistics-inputs", default=None, help="Override statistical test input CSVs.")
    parser.add_argument("--statistics-metrics", default=None, help="Metrics for statistical tests.")
    parser.add_argument("--reference-model", default="projected_constricted_nfpso", help="Reference model for statistics.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    ensure_dir(output_root)

    stages = resolve_stages(parse_csv_list(args.stages), parse_csv_list(args.skip_stages))
    missing_scripts = validate_script_files(stages)
    if missing_scripts:
        print("[ERROR] Missing required scripts:")
        for path in missing_scripts:
            print(f"  - {path}")
        return 2

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    args.csv = str(csv_path)

    if any(stage in stages for stage in ["nfpso_experiments", "real_data_benchmarks", "pso_budget"]):
        if not csv_path.exists():
            print(f"[ERROR] private CSV not found: {csv_path}")
            print("[NOTE] Use --csv data\\tax_revenues.csv or place the local authorized CSV.")
            return 2

    print("[INFO] SS-NFPSO Q1 pipeline")
    print(f"[INFO] root={ROOT}")
    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] stages={stages}")
    print(f"[INFO] quick={args.quick}")
    print(f"[INFO] csv={csv_path}")

    env = make_env()
    results: List[StageResult] = []
    pipeline_started = time.time()

    try:
        for stage in stages:
            cmd = stage_command(stage, args, output_root)
            result = run_command(
                stage=stage,
                command=cmd,
                continue_on_error=args.continue_on_error,
                env=env,
            )
            results.append(result)
    except RuntimeError:
        pipeline_finished = time.time()
        manifest_paths = write_manifest(
            results=results,
            output_dir=output_root / "q1_pipeline",
            payload={
                "status": "failed",
                "quick": args.quick,
                "root": str(ROOT),
                "output_root": str(output_root),
                "csv": str(csv_path),
                "stages": stages,
                "started_at_epoch": pipeline_started,
                "finished_at_epoch": pipeline_finished,
                "runtime_seconds": pipeline_finished - pipeline_started,
            },
        )
        print(f"[DONE] partial manifest json: {manifest_paths['manifest_json']}")
        print(f"[DONE] partial manifest csv: {manifest_paths['manifest_csv']}")
        return 1

    pipeline_finished = time.time()
    failed = [r for r in results if r.status != "ok"]
    status = "ok" if not failed else "failed"

    manifest_paths = write_manifest(
        results=results,
        output_dir=output_root / "q1_pipeline",
        payload={
            "status": status,
            "quick": args.quick,
            "root": str(ROOT),
            "output_root": str(output_root),
            "csv": str(csv_path),
            "stages": stages,
            "started_at_epoch": pipeline_started,
            "finished_at_epoch": pipeline_finished,
            "runtime_seconds": pipeline_finished - pipeline_started,
        },
    )

    print("\n[PIPELINE DONE]")
    print(f"[STATUS] {status}")
    print(f"[DONE] manifest json: {manifest_paths['manifest_json']}")
    print(f"[DONE] manifest csv: {manifest_paths['manifest_csv']}")
    print("[NOTE] outputs/ is ignored by Git and should not be committed.")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
