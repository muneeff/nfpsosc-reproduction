from __future__ import annotations

"""
Manifest-only gate for the V2 external Final benchmark.

This module intentionally cannot fit models, forecast, calculate Final accuracy,
or inspect Final outcomes.  Its only job at stage 3F-2 is to bind the already
frozen protocol, selected hyperparameters, Final seeds, real-series identities,
baseline support, software lock, Git state, and deterministic task identities
into a cryptographically protected Final execution manifest.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from typing import Literal, Sequence

import numpy as np

from .development_selection import (
    CONFIRMATORY_NOISE_LEVELS,
    DEVELOPMENT_DGP_SEEDS,
    DEVELOPMENT_OPTIMIZER_SEEDS,
    FINAL_DGP_SEEDS,
    FINAL_OPTIMIZER_SEEDS,
    LEGACY_V1_DGP_SEEDS,
)
from .synthetic import ALL_GENERATORS, seasonal_period


SCHEMA_VERSION = "v2-final-external-manifest-1"
FINAL_OUTCOME_EXECUTION_ENABLED = False

PC_METHOD = "pc_nfpso"
BASELINE_METHODS: tuple[str, ...] = (
    "naive_1",
    "seasonal_naive",
    "drift",
    "sarima",
    "ets",
    "theta",
    "ridge_lag",
    "svr_rbf",
    "xgboost",
)
METRIC_NAMES: tuple[str, ...] = ("MASE", "MAE", "RMSE", "sMAPE", "RMSSE")

EXPECTED_SELECTION_RADIUS = 1.0
EXPECTED_SELECTION_ALPHA = 0.05
EXPECTED_DEVELOPMENT_RESULT_COUNT = 28_350
EXPECTED_REAL_FINAL_SERIES = 120
EXPECTED_REAL_DEVELOPMENT_SERIES = 60
EXPECTED_SYNTHETIC_PC_TASKS = 2_700
EXPECTED_SYNTHETIC_BASELINE_TASKS = 4_440
EXPECTED_REAL_PC_TASKS = 600
EXPECTED_REAL_BASELINE_TASKS = 1_040
EXPECTED_SYNTHETIC_TASKS = 7_140
EXPECTED_REAL_TASKS = 1_640
EXPECTED_TOTAL_TASKS = 8_780

SELECTION_FREEZE_TAG = "v2-development-selection-freeze-2026-08-14"
#ebbaf56d7c0eb33044fe8633b0973811e9d43eb3
EXPECTED_SELECTION_FREEZE_TAG_TARGET = (
    "fca13e30e1429ca639e31934958f4553e9b0077d"
)

PROTOCOL_INPUT_FILES: tuple[str, ...] = (
    "configs/v2/protocol_v2.json",
    "configs/v2/protocol_freeze_sha256_v2.json",
    "configs/v2/environment_protocol_freeze_2026-08-12.txt",
    "configs/v2/synthetic_split_v2.json",
    "configs/v2/synthetic_dgps_v2.json",
    "configs/v2/canonical_synthetic_v2.json",
    "configs/v2/real_series_policy_v2.json",
    "configs/v2/real_series_manifest_v2.json",
    "configs/v2/baselines_v2.json",
    "configs/v2/metrics_v2.json",
    "configs/v2/statistics_v2.json",
    "configs/v2/ablation_v2.json",
    "configs/v2/amendment_001_aicc_tiebreaks.json",
    "configs/v2/amendment_002_sarima_convergence.json",
    "configs/v2/amendment_003_ets_convergence.json",
    "configs/v2/amendment_004_tsk_implementation.json",
    "configs/v2/amendment_005_canonical_synthetic.json",
    "configs/v2/amendment_006_lorenz_rk4_order.json",
    "configs/v2/amendment_007_pcnfpso_validation_failure.json",
    "configs/v2/amendment_008_henon_basin_rejection.json",
    "configs/v2/amendment_009_final_aggregation_metrics.json",
    "configs/v2/amendment_009_sha256.json",
    "configs/v2/development_selection_freeze_v2.json",
    "configs/v2/development_selection_freeze_sha256_v2.json",
)

SOFTWARE_LOCK: dict[str, str] = {
    "statsmodels": "0.14.6",
    "scikit-learn": "1.7.2",
    "xgboost": "3.2.0",
    "datasetsforecast": "1.0.1",
}


class FinalManifestError(RuntimeError):
    """Raised when a Final-manifest invariant is violated."""


@dataclass(frozen=True, order=True)
class FinalRunTask:
    run_id: str
    domain: Literal["synthetic", "real"]
    method: str
    generator: str | None
    dgp_seed: int | None
    noise_level: float | None
    source: str | None
    frequency: str | None
    series_id: str | None
    optimizer_seed: int | None
    seasonal_period: int
    n_lags: int
    radius: float | None
    alpha: float | None


@dataclass(frozen=True)
class FinalManifest:
    schema_version: str
    protocol_fingerprint: str
    code_commit: str
    task_count: int
    synthetic_task_count: int
    real_task_count: int
    selected_radius: float
    selected_alpha: float
    methods: tuple[str, ...]
    metrics: tuple[str, ...]
    final_dgp_seeds: tuple[int, ...]
    final_optimizer_seeds: tuple[int, ...]
    real_final_series_count: int
    tasks: tuple[FinalRunTask, ...]


@dataclass(frozen=True)
class FinalDryRunSummary:
    task_count: int
    synthetic_task_count: int
    real_task_count: int
    synthetic_pc_task_count: int
    synthetic_baseline_task_count: int
    real_pc_task_count: int
    real_baseline_task_count: int
    real_final_series_count: int
    generator_count: int
    baseline_count: int
    final_dgp_seed_count: int
    final_optimizer_seed_count: int
    selected_radius: float
    selected_alpha: float
    first_run_id: str
    last_run_id: str


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FinalManifestError(f"Missing frozen JSON file: {path.as_posix()}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalManifestError(f"Invalid JSON: {path.as_posix()}") from exc
    if not isinstance(data, dict):
        raise FinalManifestError(f"Top-level JSON object required: {path.as_posix()}")
    return data


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_hash_map(project_root: Path, relative_hash_file: str) -> None:
    path = project_root / relative_hash_file
    mapping = _read_json(path)
    for relative, expected in mapping.items():
        target = project_root / str(relative)
        if not target.is_file():
            raise FinalManifestError(
                f"Hash sidecar references missing file: {relative}"
            )
        actual = _sha256_file(target)
        if actual != str(expected):
            raise FinalManifestError(
                f"SHA-256 mismatch for {relative}: expected {expected}, got {actual}."
            )


def frozen_lag_dimension(seasonal_period_value: int) -> int:
    if isinstance(seasonal_period_value, (bool, np.bool_)) or not isinstance(
        seasonal_period_value, (int, np.integer)
    ):
        raise FinalManifestError("seasonal period must be an integer.")
    m = int(seasonal_period_value)
    if m < 1:
        raise FinalManifestError("seasonal period must be at least one.")
    return min(12, max(5, m))


def _load_selected_pair(project_root: Path) -> tuple[float, float]:
    _verify_hash_map(
        project_root,
        "configs/v2/development_selection_freeze_sha256_v2.json",
    )
    data = _read_json(
        project_root / "configs/v2/development_selection_freeze_v2.json"
    )
    if data.get("status") != "LOCKED_AFTER_COMPLETE_DEVELOPMENT_BEFORE_FINAL_BENCHMARK":
        raise FinalManifestError("Development selection freeze status is not locked.")
    results = data.get("development_results", {})
    if (
        int(results.get("record_count", -1)) != EXPECTED_DEVELOPMENT_RESULT_COUNT
        or int(results.get("successful_count", -1)) != EXPECTED_DEVELOPMENT_RESULT_COUNT
        or int(results.get("failed_count", -1)) != 0
    ):
        raise FinalManifestError(
            "Final execution requires the complete successful Development selection."
        )
    selection_rule = data.get("selection_rule", {})
    if bool(selection_rule.get("final_domain_used_for_selection", True)):
        raise FinalManifestError("Final synthetic outcomes contaminated selection.")
    if bool(selection_rule.get("real_final_domain_used_for_selection", True)):
        raise FinalManifestError("Final real outcomes contaminated selection.")
    pair = data.get("selected_hyperparameters", {})
    radius = float(pair.get("radius", np.nan))
    alpha = float(pair.get("ridge_alpha", np.nan))
    if radius != EXPECTED_SELECTION_RADIUS or alpha != EXPECTED_SELECTION_ALPHA:
        raise FinalManifestError(
            "Frozen Development-selected hyperparameters do not equal radius=1.0, alpha=0.05."
        )
    return radius, alpha


def _validate_a009(project_root: Path) -> None:
    _verify_hash_map(project_root, "configs/v2/amendment_009_sha256.json")
    data = _read_json(
        project_root / "configs/v2/amendment_009_final_aggregation_metrics.json"
    )
    if data.get("amendment_id") != "V2-A009":
        raise FinalManifestError("V2-A009 identifier mismatch.")
    if data.get("status") != "LOCKED_PRE_FINAL_PERFORMANCE":
        raise FinalManifestError("V2-A009 must be locked before Final execution.")
    trigger = data.get("trigger", {})
    if bool(trigger.get("final_performance_outcomes_generated", True)):
        raise FinalManifestError("V2-A009 does not certify a pre-Final clarification.")
    pc = data.get("pc_nfpso_final_optimizer_replicates", {})
    synthetic_seeds = tuple(
        int(v) for v in pc.get("synthetic", {}).get("optimizer_seeds", ())
    )
    real_seeds = tuple(
        int(v) for v in pc.get("real", {}).get("optimizer_seeds", ())
    )
    frozen = tuple(FINAL_OPTIMIZER_SEEDS)
    if synthetic_seeds != frozen or real_seeds != frozen:
        raise FinalManifestError("V2-A009 optimizer seed coverage mismatch.")


def _load_baseline_methods(project_root: Path) -> tuple[str, ...]:
    data = _read_json(project_root / "configs/v2/baselines_v2.json")
    if data.get("status") != "LOCKED":
        raise FinalManifestError("Baseline protocol must be LOCKED.")
    methods = tuple(str(v) for v in data.get("baselines", {}).keys())
    if methods != BASELINE_METHODS:
        raise FinalManifestError(
            f"Baseline method/order mismatch: expected {BASELINE_METHODS}, got {methods}."
        )
    seasonal = data["baselines"]["seasonal_naive"]
    if seasonal.get("availability") != "m>1 only":
        raise FinalManifestError("Seasonal-naive support rule is not frozen to m>1.")
    common = data.get("common", {})
    if not bool(common.get("no_test_outcome_for_tuning", False)):
        raise FinalManifestError("Baseline tuning firewall is not locked.")
    if not bool(common.get("no_silent_fallback", False)):
        raise FinalManifestError("Baseline no-fallback rule is not locked.")
    return methods


def _validate_metric_protocol(project_root: Path) -> None:
    data = _read_json(project_root / "configs/v2/metrics_v2.json")
    if data.get("status") != "LOCKED":
        raise FinalManifestError("Metric protocol must be LOCKED.")
    if data.get("primary_metric", {}).get("name") != "MASE":
        raise FinalManifestError("Primary metric must remain MASE.")
    secondary = tuple(str(v) for v in data.get("secondary_metrics", {}).keys())
    if secondary != METRIC_NAMES[1:]:
        raise FinalManifestError(
            f"Secondary metric/order mismatch: {secondary}."
        )
    support = data.get("seasonal_naive_support", {})
    if not bool(support.get("undefined_nonseasonal_rows_not_created", False)):
        raise FinalManifestError(
            "Nonseasonal seasonal-naive rows must remain undefined/not created."
        )


def _validate_seed_protocol(project_root: Path) -> None:
    data = _read_json(project_root / "configs/v2/synthetic_split_v2.json")
    if data.get("status") != "LOCKED":
        raise FinalManifestError("Synthetic split protocol must be LOCKED.")
    final = data.get("final_locked", {})
    dgp = tuple(int(v) for v in final.get("dgp_seeds", ()))
    opt = tuple(int(v) for v in final.get("optimizer_seeds", ()))
    if dgp != tuple(FINAL_DGP_SEEDS):
        raise FinalManifestError("Final DGP seed set/order mismatch.")
    if opt != tuple(FINAL_OPTIMIZER_SEEDS):
        raise FinalManifestError("Final optimizer seed set/order mismatch.")
    if set(dgp) & set(DEVELOPMENT_DGP_SEEDS):
        raise FinalManifestError("Development/Final DGP seed overlap detected.")
    if set(opt) & set(DEVELOPMENT_OPTIMIZER_SEEDS):
        raise FinalManifestError("Development/Final optimizer seed overlap detected.")
    if set(dgp) & set(LEGACY_V1_DGP_SEEDS):
        raise FinalManifestError("Legacy V1 DGP seed leaked into Final set.")
    levels = tuple(
        float(v)
        for v in data.get("benchmark_composition", {}).get(
            "confirmatory_noise_levels", ()
        )
    )
    if levels != tuple(CONFIRMATORY_NOISE_LEVELS):
        raise FinalManifestError("Confirmatory noise-level set/order mismatch.")


def _load_real_final_records(project_root: Path) -> tuple[dict, ...]:
    data = _read_json(project_root / "configs/v2/real_series_manifest_v2.json")
    if data.get("status") != "LOCKED":
        raise FinalManifestError("Real-series manifest must be LOCKED.")
    counts = data.get("counts", {})
    if (
        int(counts.get("development", -1)) != EXPECTED_REAL_DEVELOPMENT_SERIES
        or int(counts.get("final_locked", -1)) != EXPECTED_REAL_FINAL_SERIES
        or int(counts.get("total", -1))
        != EXPECTED_REAL_DEVELOPMENT_SERIES + EXPECTED_REAL_FINAL_SERIES
    ):
        raise FinalManifestError("Real-series manifest totals are invalid.")
    rows = tuple(data.get("selected_series", ()))
    final_rows = tuple(r for r in rows if r.get("partition") == "final_locked")
    development_rows = tuple(
        r for r in rows if r.get("partition") == "development"
    )
    if len(final_rows) != EXPECTED_REAL_FINAL_SERIES:
        raise FinalManifestError("Expected exactly 120 frozen Final real series.")
    if len(development_rows) != EXPECTED_REAL_DEVELOPMENT_SERIES:
        raise FinalManifestError("Expected exactly 60 real Development series.")
    keys = [
        (str(r["source"]), str(r["frequency"]), str(r["series_id"]))
        for r in final_rows
    ]
    if len(set(keys)) != len(keys):
        raise FinalManifestError("Duplicate Final real-series identity.")
    strata_counts: dict[tuple[str, str], int] = {}
    for row in final_rows:
        key = (str(row["source"]), str(row["frequency"]))
        strata_counts[key] = strata_counts.get(key, 0) + 1
        m = int(row["seasonal_period"])
        L = int(row["lag_L"])
        if L != frozen_lag_dimension(m):
            raise FinalManifestError(
                f"Frozen lag mismatch for real series {key}/{row['series_id']}."
            )
        if int(row["n_train"]) != int(row["split_index"]):
            raise FinalManifestError("Real-series split metadata mismatch.")
        if int(row["n_train"]) + int(row["n_test"]) != int(row["n"]):
            raise FinalManifestError("Real-series length metadata mismatch.")
        scale = float(row["mase_scale_train_only"])
        if not np.isfinite(scale) or scale <= 1e-12:
            raise FinalManifestError("Invalid frozen real-series MASE scale.")
        fingerprint = str(row["series_sha256_float64_le"])
        if len(fingerprint) != 64:
            raise FinalManifestError("Invalid real-series SHA-256 fingerprint.")
    expected_strata = {
        ("M3", "Yearly"): 20,
        ("M3", "Quarterly"): 20,
        ("M3", "Monthly"): 20,
        ("M4", "Yearly"): 20,
        ("M4", "Quarterly"): 20,
        ("M4", "Monthly"): 20,
    }
    if strata_counts != expected_strata:
        raise FinalManifestError(
            f"Final real-series stratum balance mismatch: {strata_counts}."
        )
    return final_rows


def _baseline_supported(method: str, m: int) -> bool:
    if method not in BASELINE_METHODS:
        raise FinalManifestError(f"Unknown baseline method: {method!r}")
    return method != "seasonal_naive" or int(m) > 1


def _canonical_float(value: float | None) -> str | None:
    if value is None:
        return None
    x = float(value)
    if not np.isfinite(x):
        raise FinalManifestError("Run-key float must be finite.")
    return repr(x)


def _deterministic_run_id(
    *,
    domain: str,
    method: str,
    generator: str | None,
    dgp_seed: int | None,
    noise_level: float | None,
    source: str | None,
    frequency: str | None,
    series_id: str | None,
    optimizer_seed: int | None,
    seasonal_period_value: int,
    n_lags: int,
    radius: float | None,
    alpha: float | None,
) -> str:
    payload = {
        "namespace": "pc-nfpso-v2-final-external",
        "domain": str(domain),
        "method": str(method),
        "generator": generator,
        "dgp_seed": dgp_seed,
        "noise_level": _canonical_float(noise_level),
        "source": source,
        "frequency": frequency,
        "series_id": series_id,
        "optimizer_seed": optimizer_seed,
        "seasonal_period": int(seasonal_period_value),
        "n_lags": int(n_lags),
        "radius": _canonical_float(radius),
        "alpha": _canonical_float(alpha),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _make_task(
    *,
    domain: Literal["synthetic", "real"],
    method: str,
    generator: str | None = None,
    dgp_seed: int | None = None,
    noise_level: float | None = None,
    source: str | None = None,
    frequency: str | None = None,
    series_id: str | None = None,
    optimizer_seed: int | None = None,
    seasonal_period_value: int,
    n_lags: int,
    radius: float | None = None,
    alpha: float | None = None,
) -> FinalRunTask:
    run_id = _deterministic_run_id(
        domain=domain,
        method=method,
        generator=generator,
        dgp_seed=dgp_seed,
        noise_level=noise_level,
        source=source,
        frequency=frequency,
        series_id=series_id,
        optimizer_seed=optimizer_seed,
        seasonal_period_value=seasonal_period_value,
        n_lags=n_lags,
        radius=radius,
        alpha=alpha,
    )
    return FinalRunTask(
        run_id=run_id,
        domain=domain,
        method=method,
        generator=generator,
        dgp_seed=dgp_seed,
        noise_level=noise_level,
        source=source,
        frequency=frequency,
        series_id=series_id,
        optimizer_seed=optimizer_seed,
        seasonal_period=int(seasonal_period_value),
        n_lags=int(n_lags),
        radius=radius,
        alpha=alpha,
    )


@lru_cache(maxsize=4)
def _build_final_tasks_cached(project_root_resolved: str) -> tuple[FinalRunTask, ...]:
    root = Path(project_root_resolved)
    radius, alpha = _load_selected_pair(root)
    _validate_a009(root)
    _validate_seed_protocol(root)
    methods = _load_baseline_methods(root)
    _validate_metric_protocol(root)
    real_rows = _load_real_final_records(root)

    tasks: list[FinalRunTask] = []

    for generator in ALL_GENERATORS:
        m = seasonal_period(generator)
        L = frozen_lag_dimension(m)
        for dgp_seed in FINAL_DGP_SEEDS:
            for noise_level in CONFIRMATORY_NOISE_LEVELS:
                for optimizer_seed in FINAL_OPTIMIZER_SEEDS:
                    tasks.append(
                        _make_task(
                            domain="synthetic",
                            method=PC_METHOD,
                            generator=generator,
                            dgp_seed=int(dgp_seed),
                            noise_level=float(noise_level),
                            optimizer_seed=int(optimizer_seed),
                            seasonal_period_value=m,
                            n_lags=L,
                            radius=radius,
                            alpha=alpha,
                        )
                    )
                for method in methods:
                    if not _baseline_supported(method, m):
                        continue
                    tasks.append(
                        _make_task(
                            domain="synthetic",
                            method=method,
                            generator=generator,
                            dgp_seed=int(dgp_seed),
                            noise_level=float(noise_level),
                            seasonal_period_value=m,
                            n_lags=L,
                        )
                    )

    for row in real_rows:
        source = str(row["source"])
        frequency = str(row["frequency"])
        series_id = str(row["series_id"])
        m = int(row["seasonal_period"])
        L = int(row["lag_L"])
        for optimizer_seed in FINAL_OPTIMIZER_SEEDS:
            tasks.append(
                _make_task(
                    domain="real",
                    method=PC_METHOD,
                    source=source,
                    frequency=frequency,
                    series_id=series_id,
                    optimizer_seed=int(optimizer_seed),
                    seasonal_period_value=m,
                    n_lags=L,
                    radius=radius,
                    alpha=alpha,
                )
            )
        for method in methods:
            if not _baseline_supported(method, m):
                continue
            tasks.append(
                _make_task(
                    domain="real",
                    method=method,
                    source=source,
                    frequency=frequency,
                    series_id=series_id,
                    seasonal_period_value=m,
                    n_lags=L,
                )
            )

    validate_final_tasks(tasks)
    return tuple(tasks)


def build_final_tasks(
    *, project_root: str | os.PathLike[str] = "."
) -> tuple[FinalRunTask, ...]:
    """Build the frozen Final task design without executing any task."""
    return _build_final_tasks_cached(str(Path(project_root).resolve()))


def validate_final_tasks(tasks: Sequence[FinalRunTask]) -> None:
    rows = tuple(tasks)
    if len(rows) != EXPECTED_TOTAL_TASKS:
        raise FinalManifestError(
            f"Final manifest requires exactly {EXPECTED_TOTAL_TASKS} tasks; got {len(rows)}."
        )
    run_ids = [r.run_id for r in rows]
    if len(set(run_ids)) != len(run_ids):
        raise FinalManifestError("Duplicate Final run IDs detected.")

    synthetic = tuple(r for r in rows if r.domain == "synthetic")
    real = tuple(r for r in rows if r.domain == "real")
    if len(synthetic) != EXPECTED_SYNTHETIC_TASKS:
        raise FinalManifestError("Synthetic Final task count mismatch.")
    if len(real) != EXPECTED_REAL_TASKS:
        raise FinalManifestError("Real Final task count mismatch.")

    synthetic_pc = tuple(r for r in synthetic if r.method == PC_METHOD)
    synthetic_bl = tuple(r for r in synthetic if r.method != PC_METHOD)
    real_pc = tuple(r for r in real if r.method == PC_METHOD)
    real_bl = tuple(r for r in real if r.method != PC_METHOD)
    if len(synthetic_pc) != EXPECTED_SYNTHETIC_PC_TASKS:
        raise FinalManifestError("Synthetic PC-NFPSO task count mismatch.")
    if len(synthetic_bl) != EXPECTED_SYNTHETIC_BASELINE_TASKS:
        raise FinalManifestError("Synthetic baseline task count mismatch.")
    if len(real_pc) != EXPECTED_REAL_PC_TASKS:
        raise FinalManifestError("Real PC-NFPSO task count mismatch.")
    if len(real_bl) != EXPECTED_REAL_BASELINE_TASKS:
        raise FinalManifestError("Real baseline task count mismatch.")

    real_series = {
        (r.source, r.frequency, r.series_id)
        for r in real
    }
    if len(real_series) != EXPECTED_REAL_FINAL_SERIES:
        raise FinalManifestError("Final real-series identity count mismatch.")

    for row in rows:
        if row.seasonal_period < 1:
            raise FinalManifestError("Invalid task seasonal period.")
        if row.n_lags != frozen_lag_dimension(row.seasonal_period):
            raise FinalManifestError("Task lag rule mismatch.")
        if row.method == PC_METHOD:
            if row.optimizer_seed not in FINAL_OPTIMIZER_SEEDS:
                raise FinalManifestError("PC-NFPSO task uses a non-Final optimizer seed.")
            if row.radius != EXPECTED_SELECTION_RADIUS:
                raise FinalManifestError("PC-NFPSO task radius differs from frozen selection.")
            if row.alpha != EXPECTED_SELECTION_ALPHA:
                raise FinalManifestError("PC-NFPSO task alpha differs from frozen selection.")
        else:
            if row.method not in BASELINE_METHODS:
                raise FinalManifestError("Unknown Final baseline method.")
            if row.optimizer_seed is not None:
                raise FinalManifestError("Baseline task cannot carry a PC optimizer seed.")
            if row.radius is not None or row.alpha is not None:
                raise FinalManifestError("Baseline task cannot carry PC hyperparameters.")
            if row.method == "seasonal_naive" and row.seasonal_period <= 1:
                raise FinalManifestError(
                    "Undefined nonseasonal seasonal-naive task was created."
                )

        if row.domain == "synthetic":
            if row.generator not in ALL_GENERATORS:
                raise FinalManifestError("Unexpected synthetic generator.")
            if row.dgp_seed not in FINAL_DGP_SEEDS:
                raise FinalManifestError("Synthetic task uses a non-Final DGP seed.")
            if row.noise_level not in CONFIRMATORY_NOISE_LEVELS:
                raise FinalManifestError("Synthetic task uses non-confirmatory noise.")
            if any(v is not None for v in (row.source, row.frequency, row.series_id)):
                raise FinalManifestError("Synthetic task contains real-series identity.")
        elif row.domain == "real":
            if any(v is not None for v in (row.generator, row.dgp_seed, row.noise_level)):
                raise FinalManifestError("Real task contains synthetic identity.")
            if not row.source or not row.frequency or not row.series_id:
                raise FinalManifestError("Real task lacks frozen series identity.")
        else:
            raise FinalManifestError(f"Unexpected Final domain: {row.domain!r}")

        expected_id = _deterministic_run_id(
            domain=row.domain,
            method=row.method,
            generator=row.generator,
            dgp_seed=row.dgp_seed,
            noise_level=row.noise_level,
            source=row.source,
            frequency=row.frequency,
            series_id=row.series_id,
            optimizer_seed=row.optimizer_seed,
            seasonal_period_value=row.seasonal_period,
            n_lags=row.n_lags,
            radius=row.radius,
            alpha=row.alpha,
        )
        if row.run_id != expected_id:
            raise FinalManifestError("Final task run ID mismatch.")


def final_dry_run_summary(
    *, project_root: str | os.PathLike[str] = "."
) -> FinalDryRunSummary:
    tasks = build_final_tasks(project_root=project_root)
    synthetic = tuple(r for r in tasks if r.domain == "synthetic")
    real = tuple(r for r in tasks if r.domain == "real")
    radius, alpha = _load_selected_pair(Path(project_root).resolve())
    return FinalDryRunSummary(
        task_count=len(tasks),
        synthetic_task_count=len(synthetic),
        real_task_count=len(real),
        synthetic_pc_task_count=sum(r.method == PC_METHOD for r in synthetic),
        synthetic_baseline_task_count=sum(r.method != PC_METHOD for r in synthetic),
        real_pc_task_count=sum(r.method == PC_METHOD for r in real),
        real_baseline_task_count=sum(r.method != PC_METHOD for r in real),
        real_final_series_count=len(
            {(r.source, r.frequency, r.series_id) for r in real}
        ),
        generator_count=len({r.generator for r in synthetic}),
        baseline_count=len(BASELINE_METHODS),
        final_dgp_seed_count=len({r.dgp_seed for r in synthetic}),
        final_optimizer_seed_count=len(
            {r.optimizer_seed for r in tasks if r.method == PC_METHOD}
        ),
        selected_radius=radius,
        selected_alpha=alpha,
        first_run_id=tasks[0].run_id,
        last_run_id=tasks[-1].run_id,
    )


def protocol_fingerprint(project_root: str | os.PathLike[str]) -> str:
    root = Path(project_root)
    digest = hashlib.sha256()
    for relative in PROTOCOL_INPUT_FILES:
        path = root / relative
        if not path.is_file():
            raise FinalManifestError(f"Missing frozen Final protocol input: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_head(project_root: str | os.PathLike[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FinalManifestError("Unable to resolve Git HEAD.") from exc
    head = completed.stdout.strip()
    if len(head) != 40:
        raise FinalManifestError(f"Unexpected Git HEAD: {head!r}")
    return head


def assert_clean_git_worktree(project_root: str | os.PathLike[str]) -> None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FinalManifestError("Unable to inspect Git working tree.") from exc
    if completed.stdout.strip():
        raise FinalManifestError(
            "Final manifest preparation requires a clean Git working tree."
        )


def assert_selection_freeze_tag(project_root: str | os.PathLike[str]) -> None:
    try:
        completed = subprocess.run(
            ["git", "rev-list", "-n", "1", SELECTION_FREEZE_TAG],
            cwd=Path(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FinalManifestError(
            f"Required selection-freeze tag is missing: {SELECTION_FREEZE_TAG}"
        ) from exc
    target = completed.stdout.strip()
    if target != EXPECTED_SELECTION_FREEZE_TAG_TARGET:
        raise FinalManifestError(
            "Selection-freeze tag target mismatch: "
            f"expected {EXPECTED_SELECTION_FREEZE_TAG_TARGET}, got {target}."
        )


def verify_final_software_environment() -> dict[str, str]:
    observed: dict[str, str] = {}
    for package, expected in SOFTWARE_LOCK.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise FinalManifestError(
                f"Required locked package is not installed: {package}=={expected}"
            ) from exc
        observed[package] = actual
        if actual != expected:
            raise FinalManifestError(
                f"Software lock mismatch for {package}: expected {expected}, got {actual}."
            )
    return observed


def build_final_manifest(
    *,
    project_root: str | os.PathLike[str],
    protocol_fingerprint_value: str,
    code_commit: str,
) -> FinalManifest:
    if not isinstance(protocol_fingerprint_value, str) or len(protocol_fingerprint_value) != 64:
        raise FinalManifestError("protocol_fingerprint must be a SHA-256 hex string.")
    if not isinstance(code_commit, str) or len(code_commit) != 40:
        raise FinalManifestError("code_commit must be a 40-character Git commit hash.")
    root = Path(project_root).resolve()
    radius, alpha = _load_selected_pair(root)
    tasks = build_final_tasks(project_root=root)
    return FinalManifest(
        schema_version=SCHEMA_VERSION,
        protocol_fingerprint=protocol_fingerprint_value,
        code_commit=code_commit,
        task_count=len(tasks),
        synthetic_task_count=EXPECTED_SYNTHETIC_TASKS,
        real_task_count=EXPECTED_REAL_TASKS,
        selected_radius=radius,
        selected_alpha=alpha,
        methods=(PC_METHOD,) + BASELINE_METHODS,
        metrics=METRIC_NAMES,
        final_dgp_seeds=tuple(FINAL_DGP_SEEDS),
        final_optimizer_seeds=tuple(FINAL_OPTIMIZER_SEEDS),
        real_final_series_count=EXPECTED_REAL_FINAL_SERIES,
        tasks=tasks,
    )


def _manifest_to_dict(manifest: FinalManifest) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "stage": "manifest_only_pre_final_outcomes",
        "final_outcome_execution_enabled": False,
        "protocol_fingerprint": manifest.protocol_fingerprint,
        "code_commit": manifest.code_commit,
        "task_count": manifest.task_count,
        "synthetic_task_count": manifest.synthetic_task_count,
        "real_task_count": manifest.real_task_count,
        "selected_radius": manifest.selected_radius,
        "selected_alpha": manifest.selected_alpha,
        "methods": list(manifest.methods),
        "metrics": list(manifest.metrics),
        "final_dgp_seeds": list(manifest.final_dgp_seeds),
        "final_optimizer_seeds": list(manifest.final_optimizer_seeds),
        "real_final_series_count": manifest.real_final_series_count,
        "tasks": [asdict(task) for task in manifest.tasks],
    }


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    try:
        with open(tmp, "xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_final_manifest(
    output_dir: str | os.PathLike[str],
    manifest: FinalManifest,
) -> tuple[Path, Path]:
    validate_final_tasks(manifest.tasks)
    if manifest.task_count != EXPECTED_TOTAL_TASKS:
        raise FinalManifestError("Final manifest task-count metadata mismatch.")
    if manifest.synthetic_task_count != EXPECTED_SYNTHETIC_TASKS:
        raise FinalManifestError("Final synthetic task-count metadata mismatch.")
    if manifest.real_task_count != EXPECTED_REAL_TASKS:
        raise FinalManifestError("Final real task-count metadata mismatch.")
    out = Path(output_dir)
    manifest_path = out / "final_external_manifest.json"
    hash_path = out / "final_external_manifest.sha256"
    if manifest_path.exists() or hash_path.exists():
        raise FinalManifestError("Final manifest already exists; refusing overwrite.")
    payload = _canonical_json_bytes(_manifest_to_dict(manifest))
    digest = hashlib.sha256(payload).hexdigest()
    _atomic_write_bytes(manifest_path, payload)
    try:
        _atomic_write_bytes(hash_path, (digest + "\n").encode("ascii"))
    except Exception:
        manifest_path.unlink(missing_ok=True)
        raise
    return manifest_path, hash_path


def load_final_manifest(
    output_dir: str | os.PathLike[str],
    *,
    expected_protocol_fingerprint: str | None = None,
    expected_code_commit: str | None = None,
) -> FinalManifest:
    out = Path(output_dir)
    manifest_path = out / "final_external_manifest.json"
    hash_path = out / "final_external_manifest.sha256"
    if not manifest_path.is_file() or not hash_path.is_file():
        raise FinalManifestError("Final manifest or SHA-256 sidecar is missing.")
    payload = manifest_path.read_bytes()
    expected_hash = hash_path.read_text(encoding="ascii").strip()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != expected_hash:
        raise FinalManifestError("Final manifest SHA-256 mismatch.")
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalManifestError("Final manifest is not valid JSON.") from exc
    if data.get("stage") != "manifest_only_pre_final_outcomes":
        raise FinalManifestError("Final manifest stage marker mismatch.")
    if bool(data.get("final_outcome_execution_enabled", True)):
        raise FinalManifestError("Manifest-only file illegally enables Final execution.")
    tasks = tuple(FinalRunTask(**row) for row in data["tasks"])
    manifest = FinalManifest(
        schema_version=str(data["schema_version"]),
        protocol_fingerprint=str(data["protocol_fingerprint"]),
        code_commit=str(data["code_commit"]),
        task_count=int(data["task_count"]),
        synthetic_task_count=int(data["synthetic_task_count"]),
        real_task_count=int(data["real_task_count"]),
        selected_radius=float(data["selected_radius"]),
        selected_alpha=float(data["selected_alpha"]),
        methods=tuple(str(v) for v in data["methods"]),
        metrics=tuple(str(v) for v in data["metrics"]),
        final_dgp_seeds=tuple(int(v) for v in data["final_dgp_seeds"]),
        final_optimizer_seeds=tuple(int(v) for v in data["final_optimizer_seeds"]),
        real_final_series_count=int(data["real_final_series_count"]),
        tasks=tasks,
    )
    if manifest.schema_version != SCHEMA_VERSION:
        raise FinalManifestError("Final manifest schema mismatch.")
    if manifest.methods != (PC_METHOD,) + BASELINE_METHODS:
        raise FinalManifestError("Final manifest method set/order mismatch.")
    if manifest.metrics != METRIC_NAMES:
        raise FinalManifestError("Final manifest metric set/order mismatch.")
    if manifest.final_dgp_seeds != tuple(FINAL_DGP_SEEDS):
        raise FinalManifestError("Final manifest DGP seed coverage mismatch.")
    if manifest.final_optimizer_seeds != tuple(FINAL_OPTIMIZER_SEEDS):
        raise FinalManifestError("Final manifest optimizer seed coverage mismatch.")
    if manifest.selected_radius != EXPECTED_SELECTION_RADIUS:
        raise FinalManifestError("Final manifest selected radius mismatch.")
    if manifest.selected_alpha != EXPECTED_SELECTION_ALPHA:
        raise FinalManifestError("Final manifest selected alpha mismatch.")
    if manifest.real_final_series_count != EXPECTED_REAL_FINAL_SERIES:
        raise FinalManifestError("Final manifest real-series count mismatch.")
    validate_final_tasks(manifest.tasks)
    if expected_protocol_fingerprint is not None:
        if manifest.protocol_fingerprint != expected_protocol_fingerprint:
            raise FinalManifestError("Final manifest protocol fingerprint mismatch.")
    if expected_code_commit is not None:
        if manifest.code_commit != expected_code_commit:
            raise FinalManifestError("Final manifest code commit mismatch.")
    return manifest


def prepare_final_workspace(
    output_dir: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str],
) -> FinalManifest:
    """
    Create/load the manifest-only Final workspace.

    This function performs no model fitting, forecasting, metric calculation,
    or Final outcome inspection.
    """
    if FINAL_OUTCOME_EXECUTION_ENABLED:
        raise FinalManifestError("Manifest-only stage unexpectedly enables Final execution.")
    root = Path(project_root).resolve()
    assert_clean_git_worktree(root)
    assert_selection_freeze_tag(root)
    verify_final_software_environment()
    fingerprint = protocol_fingerprint(root)
    head = git_head(root)
    out = Path(output_dir)
    manifest_path = out / "final_external_manifest.json"
    if manifest_path.exists():
        return load_final_manifest(
            out,
            expected_protocol_fingerprint=fingerprint,
            expected_code_commit=head,
        )
    manifest = build_final_manifest(
        project_root=root,
        protocol_fingerprint_value=fingerprint,
        code_commit=head,
    )
    write_final_manifest(out, manifest)
    return manifest


__all__ = [
    "SCHEMA_VERSION",
    "FINAL_OUTCOME_EXECUTION_ENABLED",
    "PC_METHOD",
    "BASELINE_METHODS",
    "METRIC_NAMES",
    "EXPECTED_SYNTHETIC_PC_TASKS",
    "EXPECTED_SYNTHETIC_BASELINE_TASKS",
    "EXPECTED_REAL_PC_TASKS",
    "EXPECTED_REAL_BASELINE_TASKS",
    "EXPECTED_SYNTHETIC_TASKS",
    "EXPECTED_REAL_TASKS",
    "EXPECTED_TOTAL_TASKS",
    "FinalManifestError",
    "FinalRunTask",
    "FinalManifest",
    "FinalDryRunSummary",
    "frozen_lag_dimension",
    "build_final_tasks",
    "validate_final_tasks",
    "final_dry_run_summary",
    "protocol_fingerprint",
    "git_head",
    "assert_clean_git_worktree",
    "assert_selection_freeze_tag",
    "verify_final_software_environment",
    "build_final_manifest",
    "write_final_manifest",
    "load_final_manifest",
    "prepare_final_workspace",
]
