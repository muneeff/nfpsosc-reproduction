"""
Run statistical tests for SS-NFPSO Q1 experiment outputs.

This script reads metrics CSV files produced by experiment runners such as:

    outputs/q1_minimal/baselines/metrics.csv
    outputs/q1_minimal/synthetic_benchmarks/metrics.csv

Then it exports model summaries, average ranks, Friedman tests, paired
reference comparisons, bootstrap confidence intervals, and Holm-corrected
p-values through `nfpsosc.statistics`.

Important
---------
- This script does not manufacture claims. It only computes evidence summaries.
- Friedman tests require at least 3 models on complete paired conditions.
- Pairwise reference comparisons require the reference model to exist in the
  input table.
- If SciPy is missing, `nfpsosc.statistics` falls back where possible.

Examples
--------
Quick test on existing quick outputs:

    py run_statistical_tests.py --quick

Use synthetic benchmark metrics only:

    py run_statistical_tests.py --inputs outputs/q1_minimal/synthetic_benchmarks/metrics.csv

Use a specific reference model:

    py run_statistical_tests.py --reference-model projected_constricted_nfpso

Use specific metrics:

    py run_statistical_tests.py --metrics rmse,mae,smape_percent,mase
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nfpsosc.statistics import (
    compare_against_reference,
    export_statistical_report,
    friedman_test,
    lower_is_better_metric,
    summarize_by_model,
)


DEFAULT_INPUTS = [
    ROOT / "outputs" / "q1_minimal" / "baselines" / "metrics.csv",
    ROOT / "outputs" / "q1_minimal" / "synthetic_benchmarks" / "metrics.csv",
]

DEFAULT_METRICS = [
    "rmse",
    "mae",
    "smape_percent",
    "mase",
]


def parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated CLI list."""

    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_path(path_text: str) -> Path:
    """Return absolute path relative to project root when needed."""

    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def ensure_dir(path: Path) -> Path:
    """Create directory if needed."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def read_metrics(path: Path) -> Optional[pd.DataFrame]:
    """Read metrics CSV if it exists and contains usable rows."""

    if not path.exists():
        print(f"[SKIP] input not found: {path}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[SKIP] failed to read {path}: {type(exc).__name__}: {exc}")
        return None

    if df.empty:
        print(f"[SKIP] empty metrics file: {path}")
        return None

    if "model" not in df.columns:
        print(f"[SKIP] no model column in: {path}")
        return None

    df = df.copy()
    df["source_file"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

    if "dataset" not in df.columns:
        # Use row index as a weak fallback. Better inputs should have dataset.
        df["dataset"] = [f"row_{i}" for i in range(len(df))]

    if "status" in df.columns:
        before = len(df)
        df = df[df["status"].fillna("ok").astype(str).str.lower().eq("ok")].copy()
        removed = before - len(df)
        if removed:
            print(f"[INFO] removed failed rows from {path.name}: {removed}")

    if df.empty:
        print(f"[SKIP] no successful rows in: {path}")
        return None

    print(f"[LOAD] {path} rows={len(df)} models={sorted(df['model'].dropna().astype(str).unique())}")
    return df


def detect_condition_cols(df: pd.DataFrame, forced: Optional[Sequence[str]] = None) -> List[str]:
    """
    Detect paired condition columns.

    Default is intentionally simple and robust: use dataset.
    For combined files, source_file is included to avoid accidental collisions.
    """

    if forced:
        missing = [col for col in forced if col not in df.columns]
        if missing:
            raise ValueError(f"Requested condition columns are missing: {missing}")
        return list(forced)

    if "source_file" in df.columns and df["source_file"].nunique() > 1:
        return ["source_file", "dataset"]

    return ["dataset"]


def available_metrics(df: pd.DataFrame, requested: Sequence[str]) -> List[str]:
    """Return numeric metrics available in the data."""

    metrics: List[str] = []
    for metric in requested:
        if metric not in df.columns:
            print(f"[SKIP] metric not found: {metric}")
            continue
        values = pd.to_numeric(df[metric], errors="coerce")
        if values.notna().sum() == 0:
            print(f"[SKIP] metric has no numeric values: {metric}")
            continue
        metrics.append(metric)
    return metrics


def model_count_by_condition(df: pd.DataFrame, condition_cols: Sequence[str]) -> pd.DataFrame:
    """Return number of models per paired condition."""

    return (
        df.dropna(subset=["model"])
        .groupby(list(condition_cols), as_index=False)["model"]
        .nunique()
        .rename(columns={"model": "n_models"})
    )


def choose_reference_model(
    df: pd.DataFrame,
    metric: str,
    condition_cols: Sequence[str],
    requested_reference: Optional[str],
) -> Optional[str]:
    """
    Choose a reference model.

    Priority:
    1. requested reference if present;
    2. projected_constricted_nfpso if present;
    3. legacy_nfpso if present;
    4. None.

    We do not automatically choose the best baseline as reference because that
    can distort interpretation in a manuscript.
    """

    models = set(df["model"].dropna().astype(str).unique())

    if requested_reference:
        if requested_reference in models:
            return requested_reference
        print(f"[SKIP] reference model not found for {metric}: {requested_reference}")
        return None

    for candidate in ["projected_constricted_nfpso", "legacy_nfpso"]:
        if candidate in models:
            return candidate

    return None


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON manifest."""

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_reports_for_frame(
    df: pd.DataFrame,
    name: str,
    output_root: Path,
    metrics: Sequence[str],
    reference_model: Optional[str],
    condition_cols: Optional[Sequence[str]],
    bootstrap_resamples: int,
    alpha: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """Run statistical reports for one DataFrame."""

    report_rows: List[Dict[str, Any]] = []
    out_dir = ensure_dir(output_root / name)

    detected_condition_cols = detect_condition_cols(df, forced=condition_cols)
    condition_counts = model_count_by_condition(df, detected_condition_cols)
    condition_counts.to_csv(out_dir / "condition_model_counts.csv", index=False, encoding="utf-8-sig")

    metrics_to_run = available_metrics(df, metrics)
    if not metrics_to_run:
        print(f"[SKIP] no requested metrics available for {name}")
        return report_rows

    print(f"[INFO] report={name} condition_cols={detected_condition_cols}")

    for metric in metrics_to_run:
        metric_dir = ensure_dir(out_dir / metric)
        ref = choose_reference_model(
            df,
            metric=metric,
            condition_cols=detected_condition_cols,
            requested_reference=reference_model,
        )

        print(f"[STAT] {name} metric={metric} reference={ref or 'none'}")

        try:
            paths = export_statistical_report(
                df,
                output_dir=metric_dir,
                metric=metric,
                reference_model=ref,
                model_col="model",
                condition_cols=detected_condition_cols,
                alpha=alpha,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
            )
            status = "ok"
            error = None
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            paths = {}

        report_rows.append(
            {
                "report": name,
                "metric": metric,
                "reference_model": ref,
                "condition_cols": ",".join(detected_condition_cols),
                "lower_is_better": lower_is_better_metric(metric),
                "status": status,
                "error": error,
                **{f"path_{key}": value for key, value in paths.items()},
            }
        )

    return report_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run statistical tests on experiment metrics.")
    parser.add_argument(
        "--inputs",
        default=None,
        help="Comma-separated metrics CSV files. Defaults to baselines and synthetic benchmark metrics.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/q1_minimal/statistical_tests",
        help="Output directory for statistical reports.",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics to analyze.",
    )
    parser.add_argument(
        "--reference-model",
        default=None,
        help="Reference model for paired comparisons, e.g. projected_constricted_nfpso.",
    )
    parser.add_argument(
        "--condition-cols",
        default=None,
        help="Comma-separated condition columns. Default: dataset, or source_file+dataset for combined files.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10000,
        help="Bootstrap resamples for paired mean-difference confidence intervals.",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for bootstrap.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer bootstrap resamples for a fast smoke test.",
    )
    args = parser.parse_args()

    input_paths = (
        [normalize_path(p) for p in parse_csv_list(args.inputs)]
        if args.inputs
        else DEFAULT_INPUTS
    )
    metrics = parse_csv_list(args.metrics) or DEFAULT_METRICS
    condition_cols = parse_csv_list(args.condition_cols)

    bootstrap_resamples = min(args.bootstrap_resamples, 200) if args.quick else args.bootstrap_resamples

    output_root = ensure_dir(normalize_path(args.output_dir))
    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] metrics={metrics}")
    print(f"[INFO] bootstrap_resamples={bootstrap_resamples}")

    frames: List[pd.DataFrame] = []
    named_frames: Dict[str, pd.DataFrame] = {}

    for path in input_paths:
        frame = read_metrics(path)
        if frame is None:
            continue
        frames.append(frame)
        safe_name = path.parent.name if path.parent.name else path.stem
        # Avoid overwriting duplicate names.
        if safe_name in named_frames:
            safe_name = f"{safe_name}_{path.stem}"
        named_frames[safe_name] = frame

    if not frames:
        print("[ERROR] No usable metrics files found.")
        return 2

    all_report_rows: List[Dict[str, Any]] = []

    for name, frame in named_frames.items():
        all_report_rows.extend(
            run_reports_for_frame(
                frame,
                name=name,
                output_root=output_root,
                metrics=metrics,
                reference_model=args.reference_model,
                condition_cols=condition_cols,
                bootstrap_resamples=bootstrap_resamples,
                alpha=args.alpha,
                seed=args.seed,
            )
        )

    if len(frames) > 1:
        combined = pd.concat(frames, ignore_index=True)
        all_report_rows.extend(
            run_reports_for_frame(
                combined,
                name="combined",
                output_root=output_root,
                metrics=metrics,
                reference_model=args.reference_model,
                condition_cols=None if condition_cols is None else condition_cols,
                bootstrap_resamples=bootstrap_resamples,
                alpha=args.alpha,
                seed=args.seed,
            )
        )

    manifest_df = pd.DataFrame(all_report_rows)
    manifest_csv = output_root / "statistical_tests_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False, encoding="utf-8-sig")

    manifest_json = output_root / "statistical_tests_manifest.json"
    write_manifest(
        manifest_json,
        {
            "inputs": [str(p) for p in input_paths],
            "output_root": str(output_root),
            "metrics": metrics,
            "reference_model": args.reference_model,
            "condition_cols": condition_cols,
            "bootstrap_resamples": bootstrap_resamples,
            "alpha": args.alpha,
            "seed": args.seed,
            "reports": all_report_rows,
            "notes": [
                "Friedman tests require at least three models on complete paired conditions.",
                "Reference comparisons are skipped unless the requested/default reference exists.",
                "Generated statistical outputs are evidence summaries, not automatic publication claims.",
            ],
        },
    )

    print(f"[DONE] manifest csv: {manifest_csv}")
    print(f"[DONE] manifest json: {manifest_json}")
    print("[NOTE] outputs/ is ignored by Git and should not be committed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
