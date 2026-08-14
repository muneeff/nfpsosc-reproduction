from __future__ import annotations

"""
V2 Final external execution core.

This module executes ONLY tasks already frozen in
outputs/v2/final_external/final_external_manifest.json.  It never constructs a
new Final design and never accepts radius/alpha overrides.

The manifest was frozen before Final outcomes at commit c56504b and tagged via
the subsequent manifest-freeze commit.  Execution code may be committed later,
but it must remain a descendant of that freeze commit and the frozen protocol
fingerprint/manifest SHA must remain unchanged.
"""

from dataclasses import asdict, dataclass, is_dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from . import final_runner as fr
from .baselines import accuracy_arrays, run_baseline_v2
from .development_runner import pc_nfpso_validation_size
from .pc_nfpso import PCNFPSOFit, fit_pc_nfpso_v2, forecast_pc_nfpso_v2
from .synthetic import SERIES_LENGTH, generate_synthetic_series


FINAL_MANIFEST_FREEZE_TAG = "v2-final-external-manifest-freeze-2026-08-14"
FINAL_MANIFEST_FREEZE_COMMIT = "381288a72157cad2ecaada6618cbdbed18669b5b"
FROZEN_MANIFEST_CODE_COMMIT = "c56504bc8d95ebac9aa6e51c165769903077f348"
FROZEN_PROTOCOL_FINGERPRINT = "545f5be0ec0d14b724ed6dd76189019ab78a72acb39fec0ed53878e2d67700c8"
FROZEN_MANIFEST_SHA256 = "763da2f1e18a1f2572e97f7ece8b17438db0e20c3fe04ac0d05980208f28d5fd"
FROZEN_TOTAL_TASKS = 8_780
SYNTHETIC_PRETEST_LENGTH = 144
SYNTHETIC_TEST_LENGTH = 36
MIN_MASE_SCALE = 1e-12
MIN_RMSSE_SCALE_SQ = 1e-24
RESULT_SCHEMA = "v2-final-external-result-1"


class FinalExecutionError(RuntimeError):
    """Raised when Final execution violates the frozen execution contract."""


@dataclass(frozen=True)
class FinalMetrics:
    mase: float
    mae: float
    rmse: float
    smape: float
    rmsse: float
    mase_scale: float
    rmsse_scale_sq: float


@dataclass(frozen=True)
class FinalRunFailure:
    stage: Literal["load", "fit", "forecast", "baseline", "metrics"]
    error_type: str
    message: str
    component_stage: str | None = None
    origin: int | None = None


@dataclass(frozen=True)
class FinalRunResult:
    task: fr.FinalRunTask
    status: Literal["success", "failed"]
    metrics: FinalMetrics | None
    y_true: tuple[float, ...]
    y_pred: tuple[float, ...]
    validation_size: int | None
    best_objective: float | None
    n_rules: int | None
    best_position_sha256: str | None
    selected_config: dict[str, Any]
    selection_failures: tuple[dict[str, Any], ...]
    failure: FinalRunFailure | None

    @property
    def successful(self) -> bool:
        return self.status == "success"


@dataclass(frozen=True)
class FinalExecutionPreflight:
    execution_code_commit: str
    manifest_code_commit: str
    protocol_fingerprint: str
    manifest_sha256: str
    task_count: int
    selected_radius: float
    selected_alpha: float


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FinalExecutionError(f"Missing required JSON file: {path.as_posix()}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise FinalExecutionError(f"Top-level JSON object required: {path.as_posix()}")
    return data


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git(project_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FinalExecutionError(f"Git command failed: {' '.join(args)}") from exc
    return completed.stdout.strip()


def _assert_clean_worktree(project_root: Path) -> None:
    if _git(project_root, "status", "--porcelain"):
        raise FinalExecutionError("Final execution requires a clean Git working tree.")


def _assert_freeze_tag_and_ancestry(project_root: Path) -> str:
    target = _git(project_root, "rev-list", "-n", "1", FINAL_MANIFEST_FREEZE_TAG)
    if target != FINAL_MANIFEST_FREEZE_COMMIT:
        raise FinalExecutionError(
            f"Final manifest freeze tag target mismatch: {target!r}."
        )
    head = _git(project_root, "rev-parse", "HEAD")
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", FINAL_MANIFEST_FREEZE_COMMIT, head],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise FinalExecutionError(
            "Execution HEAD is not a descendant of the frozen Final manifest commit."
        ) from exc
    return head


def _verify_final_freeze_metadata(project_root: Path) -> dict[str, Any]:
    hash_map = _read_json(
        project_root / "configs/v2/final_external_manifest_freeze_sha256_v2.json"
    )
    for relative, expected in hash_map.items():
        target = project_root / str(relative)
        if not target.is_file():
            raise FinalExecutionError(f"Freeze hash sidecar references missing file: {relative}")
        actual = _sha256_file(target)
        if actual != str(expected):
            raise FinalExecutionError(
                f"Final freeze metadata SHA-256 mismatch for {relative}."
            )

    freeze = _read_json(
        project_root / "configs/v2/final_external_manifest_freeze_v2.json"
    )
    if freeze.get("status") != "LOCKED_PRE_FINAL_OUTCOME_EXECUTION":
        raise FinalExecutionError("Final external manifest freeze is not locked.")
    if str(freeze.get("code_commit")) != FROZEN_MANIFEST_CODE_COMMIT:
        raise FinalExecutionError("Frozen manifest code commit mismatch.")
    if str(freeze.get("protocol_fingerprint")) != FROZEN_PROTOCOL_FINGERPRINT:
        raise FinalExecutionError("Frozen protocol fingerprint mismatch.")
    manifest_meta = freeze.get("final_external_manifest", {})
    if str(manifest_meta.get("sha256")) != FROZEN_MANIFEST_SHA256:
        raise FinalExecutionError("Frozen Final manifest SHA-256 metadata mismatch.")
    design = freeze.get("design", {})
    if int(design.get("task_count_total", -1)) != FROZEN_TOTAL_TASKS:
        raise FinalExecutionError("Frozen Final task-count metadata mismatch.")
    outcomes = freeze.get("outcome_state", {})
    if bool(outcomes.get("final_outcome_execution_enabled_at_manifest_creation", True)):
        raise FinalExecutionError("Freeze metadata does not certify pre-outcome manifest creation.")
    if bool(outcomes.get("final_outcomes_generated_at_manifest_creation", True)):
        raise FinalExecutionError("Freeze metadata does not certify zero Final outcomes at freeze.")
    return freeze


def _verify_real_source_snapshot(project_root: Path) -> None:
    doc = _read_json(project_root / "data/v2_real_manifest/source_sha256_v2.json")
    if str(doc.get("datasetsforecast_version")) != "1.0.1":
        raise FinalExecutionError("Frozen real-data loader version metadata mismatch.")
    relative_root = str(doc.get("source_root"))
    if relative_root != "data/v2_real_manifest/sources":
        raise FinalExecutionError("Frozen real-data source-root metadata mismatch.")
    source_root = project_root / relative_root
    files = doc.get("files", ())
    if len(files) != 15:
        raise FinalExecutionError("Frozen real-data source hash manifest must contain 15 files.")
    for row in files:
        path = source_root / str(row["path"])
        if not path.is_file():
            raise FinalExecutionError(f"Frozen real-data source file is missing: {row['path']}")
        if path.stat().st_size != int(row["size_bytes"]):
            raise FinalExecutionError(f"Frozen real-data source size mismatch: {row['path']}")
        if _sha256_file(path) != str(row["sha256"]):
            raise FinalExecutionError(f"Frozen real-data source SHA-256 mismatch: {row['path']}")


def verify_final_execution_preflight(
    workspace: str | os.PathLike[str] = r"outputs\v2\final_external",
    *,
    project_root: str | os.PathLike[str] = ".",
) -> FinalExecutionPreflight:
    root = Path(project_root).resolve()
    _assert_clean_worktree(root)
    execution_head = _assert_freeze_tag_and_ancestry(root)
    freeze = _verify_final_freeze_metadata(root)

    if fr.protocol_fingerprint(root) != FROZEN_PROTOCOL_FINGERPRINT:
        raise FinalExecutionError("Frozen protocol fingerprint drift detected.")
    fr.verify_final_software_environment()
    _verify_real_source_snapshot(root)

    out = Path(workspace)
    manifest_path = out / "final_external_manifest.json"
    sidecar_path = out / "final_external_manifest.sha256"
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise FinalExecutionError("Frozen Final manifest or SHA-256 sidecar is missing.")
    observed = _sha256_file(manifest_path)
    sidecar = sidecar_path.read_text(encoding="ascii").strip()
    if observed != FROZEN_MANIFEST_SHA256 or sidecar != FROZEN_MANIFEST_SHA256:
        raise FinalExecutionError("Frozen Final manifest bytes no longer match the freeze.")
    manifest = fr.load_final_manifest(
        out,
        expected_protocol_fingerprint=FROZEN_PROTOCOL_FINGERPRINT,
        expected_code_commit=FROZEN_MANIFEST_CODE_COMMIT,
    )
    if manifest.task_count != FROZEN_TOTAL_TASKS:
        raise FinalExecutionError("Frozen Final manifest task count changed.")
    return FinalExecutionPreflight(
        execution_code_commit=execution_head,
        manifest_code_commit=manifest.code_commit,
        protocol_fingerprint=manifest.protocol_fingerprint,
        manifest_sha256=observed,
        task_count=manifest.task_count,
        selected_radius=manifest.selected_radius,
        selected_alpha=manifest.selected_alpha,
    )


def _finite_1d(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < 1 or not np.all(np.isfinite(arr)):
        raise FinalExecutionError(f"{name} must be non-empty and finite.")
    return arr


def compute_final_metrics(
    training: Sequence[float] | np.ndarray,
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    *,
    seasonal_period: int,
) -> FinalMetrics:
    train = _finite_1d(training, name="training")
    y = _finite_1d(actual, name="actual")
    p = _finite_1d(predicted, name="predicted")
    if y.shape != p.shape:
        raise FinalExecutionError("Actual and predicted arrays must have equal shape.")
    m = int(seasonal_period)
    if m < 1 or train.size <= m:
        raise FinalExecutionError("Invalid seasonal period for scale metrics.")

    diff = train[m:] - train[:-m]
    mase_scale = float(np.mean(np.abs(diff)))
    rmsse_scale_sq = float(np.mean(diff * diff))
    if not np.isfinite(mase_scale) or mase_scale <= MIN_MASE_SCALE:
        raise FinalExecutionError("MASE denominator is invalid.")
    if not np.isfinite(rmsse_scale_sq) or rmsse_scale_sq <= MIN_RMSSE_SCALE_SQ:
        raise FinalExecutionError("RMSSE denominator is invalid.")

    err = y - p
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    mase = float(mae / mase_scale)

    denominator = np.abs(y) + np.abs(p)
    terms = np.zeros_like(denominator, dtype=float)
    nonzero = denominator > 0.0
    terms[nonzero] = np.abs(err[nonzero]) / denominator[nonzero]
    smape = float(200.0 * np.mean(terms))
    rmsse = float(np.sqrt(np.mean(err * err) / rmsse_scale_sq))

    values = (mase, mae, rmse, smape, rmsse, mase_scale, rmsse_scale_sq)
    if not all(np.isfinite(v) and v >= 0.0 for v in values):
        raise FinalExecutionError("Computed metric bundle is invalid.")
    return FinalMetrics(
        mase=mase,
        mae=mae,
        rmse=rmse,
        smape=smape,
        rmsse=rmsse,
        mase_scale=mase_scale,
        rmsse_scale_sq=rmsse_scale_sq,
    )


def _series_sha256(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype="<f8")
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


@lru_cache(maxsize=1)
def _real_manifest_records(project_root_resolved: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    root = Path(project_root_resolved)
    data = _read_json(root / "configs/v2/real_series_manifest_v2.json")
    records = {}
    for row in data.get("selected_series", ()):
        if row.get("partition") != "final_locked":
            continue
        key = (str(row["source"]), str(row["frequency"]), str(row["series_id"]))
        records[key] = dict(row)
    if len(records) != 120:
        raise FinalExecutionError("Expected exactly 120 frozen Final real-series records.")
    return records


@lru_cache(maxsize=6)
def _load_real_group(
    project_root_resolved: str,
    source: str,
    frequency: str,
):
    root = Path(project_root_resolved)
    source_root = root / "data/v2_real_manifest/sources"
    if source == "M3":
        from datasetsforecast.m3 import M3
        frame, *_ = M3.load(directory=str(source_root), group=frequency)
    elif source == "M4":
        from datasetsforecast.m4 import M4
        frame, *_ = M4.load(directory=str(source_root), group=frequency, cache=False)
    else:
        raise FinalExecutionError(f"Unsupported real-data source: {source!r}.")
    required = {"unique_id", "y"}
    if not required.issubset(frame.columns):
        raise FinalExecutionError(f"{source}/{frequency} loader columns are invalid.")
    out = frame[["unique_id", "y"]].copy()
    out["unique_id"] = out["unique_id"].astype(str)
    out["y"] = np.asarray(out["y"], dtype=float)
    return out


def load_frozen_real_task_series(
    task: fr.FinalRunTask,
    *,
    project_root: str | os.PathLike[str] = ".",
) -> tuple[np.ndarray, np.ndarray]:
    if task.domain != "real":
        raise FinalExecutionError("Real-series loader received a non-real task.")
    if not task.source or not task.frequency or not task.series_id:
        raise FinalExecutionError("Real task identity is incomplete.")
    root = Path(project_root).resolve()
    key = (task.source, task.frequency, task.series_id)
    record = _real_manifest_records(str(root)).get(key)
    if record is None:
        raise FinalExecutionError("Real task is not one of the 120 frozen Final series.")
    frame = _load_real_group(str(root), task.source, task.frequency)
    mask = frame["unique_id"] == task.series_id
    if int(mask.sum()) < 1:
        raise FinalExecutionError("Frozen real series is missing from the source snapshot.")
    y = frame.loc[mask, "y"].to_numpy(dtype=float)
    if not np.all(np.isfinite(y)):
        raise FinalExecutionError("Frozen real series contains non-finite values.")
    if _series_sha256(y) != str(record["series_sha256_float64_le"]):
        raise FinalExecutionError("Frozen real-series fingerprint mismatch.")
    if len(y) != int(record["n"]):
        raise FinalExecutionError("Frozen real-series length mismatch.")
    split = int(record["split_index"])
    if split != int(record["n_train"]) or len(y) - split != int(record["n_test"]):
        raise FinalExecutionError("Frozen real-series split metadata mismatch.")
    if int(record["seasonal_period"]) != task.seasonal_period:
        raise FinalExecutionError("Frozen real-series seasonal-period mismatch.")
    if int(record["lag_L"]) != task.n_lags:
        raise FinalExecutionError("Frozen real-series lag mismatch.")
    train = y[:split].copy()
    test = y[split:].copy()
    metrics_scale = compute_final_metrics(
        train, test, test, seasonal_period=task.seasonal_period
    ).mase_scale
    if metrics_scale != float(record["mase_scale_train_only"]):
        raise FinalExecutionError("Frozen real-series MASE scale mismatch.")
    return train, test


def load_final_task_series(
    task: fr.FinalRunTask,
    *,
    project_root: str | os.PathLike[str] = ".",
) -> tuple[np.ndarray, np.ndarray]:
    if task.domain == "synthetic":
        if task.generator is None or task.dgp_seed is None or task.noise_level is None:
            raise FinalExecutionError("Synthetic task identity is incomplete.")
        series = generate_synthetic_series(
            task.generator,
            dgp_seed=int(task.dgp_seed),
            noise_level=float(task.noise_level),
        )
        observed = np.asarray(series.observed, dtype=float).reshape(-1)
        if observed.shape != (SERIES_LENGTH,) or SERIES_LENGTH != 180:
            raise FinalExecutionError("Frozen synthetic series length mismatch.")
        if not np.all(np.isfinite(observed)):
            raise FinalExecutionError("Synthetic Final series is non-finite.")
        train = observed[:SYNTHETIC_PRETEST_LENGTH].copy()
        test = observed[SYNTHETIC_PRETEST_LENGTH:].copy()
        if len(train) != SYNTHETIC_PRETEST_LENGTH or len(test) != SYNTHETIC_TEST_LENGTH:
            raise FinalExecutionError("Frozen synthetic 80/20 split mismatch.")
        return train, test
    if task.domain == "real":
        return load_frozen_real_task_series(task, project_root=project_root)
    raise FinalExecutionError(f"Unsupported Final task domain: {task.domain!r}.")


@lru_cache(maxsize=4)
def _frozen_task_map(project_root_resolved: str) -> dict[str, fr.FinalRunTask]:
    tasks = fr.build_final_tasks(project_root=project_root_resolved)
    fr.validate_final_tasks(tasks)
    mapping = {task.run_id: task for task in tasks}
    if len(mapping) != FROZEN_TOTAL_TASKS:
        raise FinalExecutionError("Frozen Final task map count mismatch.")
    return mapping


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FinalExecutionError("Non-finite float cannot be serialized.")
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _best_position_sha256(fitted: PCNFPSOFit) -> str:
    arr = np.asarray(fitted.optimizer_result.best_position, dtype="<f8").reshape(-1)
    if arr.size < 1 or not np.all(np.isfinite(arr)):
        raise FinalExecutionError("PC-NFPSO best position is invalid.")
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def _failed(
    task: fr.FinalRunTask,
    stage: str,
    exc: Exception,
    *,
    y_true: Sequence[float] = (),
    y_pred: Sequence[float] = (),
    validation_size: int | None = None,
    selected_config: Mapping[str, Any] | None = None,
    selection_failures: Sequence[Any] = (),
    component_stage: str | None = None,
    origin: int | None = None,
) -> FinalRunResult:
    return FinalRunResult(
        task=task,
        status="failed",
        metrics=None,
        y_true=tuple(float(v) for v in y_true),
        y_pred=tuple(float(v) for v in y_pred),
        validation_size=validation_size,
        best_objective=None,
        n_rules=None,
        best_position_sha256=None,
        selected_config=dict(_json_safe(dict(selected_config or {}))),
        selection_failures=tuple(
            safe if isinstance(safe := _json_safe(v), dict) else {"value": safe}
            for v in selection_failures
        ),
        failure=FinalRunFailure(
            stage=stage,  # type: ignore[arg-type]
            error_type=type(exc).__name__,
            message=str(exc),
            component_stage=component_stage,
            origin=origin,
        ),
    )


def execute_final_task(
    task: fr.FinalRunTask,
    *,
    project_root: str | os.PathLike[str] = ".",
    fit_function: Callable[..., PCNFPSOFit] = fit_pc_nfpso_v2,
    forecast_function: Callable[[PCNFPSOFit, np.ndarray], np.ndarray] = forecast_pc_nfpso_v2,
    baseline_function: Callable[..., Any] = run_baseline_v2,
    series_loader: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
) -> FinalRunResult:
    # Validate the task identity independently; production scheduling additionally
    # proves membership in the frozen manifest.
    frozen = _frozen_task_map(str(Path(project_root).resolve())).get(task.run_id)
    if frozen != task:
        raise FinalExecutionError("Task is not an exact member of the frozen Final manifest design.")

    loader = load_final_task_series if series_loader is None else series_loader
    try:
        train, test = loader(task, project_root=project_root)
        train = _finite_1d(train, name="training")
        test = _finite_1d(test, name="test")
    except Exception as exc:
        return _failed(task, "load", exc)

    validation_size: int | None = None
    selected_config: dict[str, Any] = {}
    selection_failures: tuple[dict[str, Any], ...] = ()

    if task.method == fr.PC_METHOD:
        validation_size = pc_nfpso_validation_size(len(train), task.n_lags)
        try:
            fitted = fit_function(
                train,
                n_lags=task.n_lags,
                validation_size=validation_size,
                radius=task.radius,
                alpha=task.alpha,
                optimizer_seed=task.optimizer_seed,
            )
        except Exception as exc:
            return _failed(
                task, "fit", exc, y_true=test, validation_size=validation_size
            )
        try:
            predicted = np.asarray(
                forecast_function(fitted, test), dtype=float
            ).reshape(-1)
            if predicted.shape != test.shape or not np.all(np.isfinite(predicted)):
                raise FinalExecutionError("PC-NFPSO Final forecast is invalid.")
        except Exception as exc:
            return _failed(
                task, "forecast", exc, y_true=test, validation_size=validation_size
            )
        try:
            metrics = compute_final_metrics(
                train, test, predicted, seasonal_period=task.seasonal_period
            )
            best_objective = float(fitted.optimizer_result.best_cost)
            n_rules = int(fitted.final_model.n_rules)
            if not np.isfinite(best_objective) or n_rules < 1:
                raise FinalExecutionError("PC-NFPSO fit provenance is invalid.")
            best_position_hash = _best_position_sha256(fitted)
        except Exception as exc:
            return _failed(
                task,
                "metrics",
                exc,
                y_true=test,
                y_pred=predicted,
                validation_size=validation_size,
            )
        pc_config = {
            "dynamics": str(fitted.dynamics),
            "boundary": str(fitted.boundary),
            "objective_components": _json_safe(fitted.selected_candidate.components),
        }
        result = FinalRunResult(
            task=task,
            status="success",
            metrics=metrics,
            y_true=tuple(float(v) for v in test),
            y_pred=tuple(float(v) for v in predicted),
            validation_size=validation_size,
            best_objective=best_objective,
            n_rules=n_rules,
            best_position_sha256=best_position_hash,
            selected_config=dict(pc_config),
            selection_failures=(),
            failure=None,
        )
    else:
        try:
            baseline_result = baseline_function(
                train,
                test,
                model=task.method,
                seasonal_period=task.seasonal_period,
            )
            selected_config = dict(_json_safe(baseline_result.selected_config))
            selection_failures = tuple(
                dict(_json_safe(v)) for v in baseline_result.selection_failures
            )
            if baseline_result.status != "success":
                failure = baseline_result.failure
                if failure is None:
                    return _failed(
                        task,
                        "baseline",
                        FinalExecutionError("Baseline returned failed status without provenance."),
                        y_true=test,
                        selected_config=selected_config,
                        selection_failures=selection_failures,
                    )
                return _failed(
                    task,
                    "baseline",
                    FinalExecutionError(failure.message),
                    y_true=test,
                    selected_config=selected_config,
                    selection_failures=selection_failures,
                    component_stage=str(failure.stage),
                    origin=None if failure.origin is None else int(failure.origin),
                )
            y_true, predicted = accuracy_arrays(baseline_result)
        except Exception as exc:
            return _failed(
                task,
                "baseline",
                exc,
                y_true=test,
                selected_config=selected_config,
                selection_failures=selection_failures,
            )
        try:
            metrics = compute_final_metrics(
                train, y_true, predicted, seasonal_period=task.seasonal_period
            )
        except Exception as exc:
            return _failed(
                task,
                "metrics",
                exc,
                y_true=y_true,
                y_pred=predicted,
                selected_config=selected_config,
                selection_failures=selection_failures,
            )
        result = FinalRunResult(
            task=task,
            status="success",
            metrics=metrics,
            y_true=tuple(float(v) for v in y_true),
            y_pred=tuple(float(v) for v in predicted),
            validation_size=None,
            best_objective=None,
            n_rules=None,
            best_position_sha256=None,
            selected_config=selected_config,
            selection_failures=selection_failures,
            failure=None,
        )

    validate_final_result(result)
    return result


def validate_final_result(result: FinalRunResult) -> None:
    task = result.task
    frozen = _frozen_task_map(str(Path(".").resolve())).get(task.run_id)
    if frozen != task:
        raise FinalExecutionError("Result task is not in the frozen Final design.")
    if result.status == "success":
        if result.failure is not None or result.metrics is None:
            raise FinalExecutionError("Successful Final result has inconsistent status/provenance.")
        y = np.asarray(result.y_true, dtype=float)
        p = np.asarray(result.y_pred, dtype=float)
        if y.shape != p.shape or y.size < 1 or not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
            raise FinalExecutionError("Successful Final result contains invalid prediction arrays.")
        for value in asdict(result.metrics).values():
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise FinalExecutionError("Successful Final result contains an invalid metric.")
        if task.method == fr.PC_METHOD:
            if result.validation_size is None or result.best_objective is None or result.n_rules is None:
                raise FinalExecutionError("Successful PC-NFPSO result lacks fit provenance.")
            if result.best_position_sha256 is None or len(result.best_position_sha256) != 64:
                raise FinalExecutionError("Successful PC-NFPSO result lacks best-position hash.")
        else:
            if result.validation_size is not None or result.best_objective is not None or result.n_rules is not None:
                raise FinalExecutionError("Baseline result illegally contains PC-NFPSO provenance.")
    elif result.status == "failed":
        if result.failure is None or result.metrics is not None:
            raise FinalExecutionError("Failed Final result has inconsistent failure provenance.")
    else:
        raise FinalExecutionError(f"Invalid Final result status: {result.status!r}.")


def _task_dict(task: fr.FinalRunTask) -> dict[str, Any]:
    return asdict(task)


def _result_dict(result: FinalRunResult) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA,
        "task": _task_dict(result.task),
        "status": result.status,
        "metrics": None if result.metrics is None else asdict(result.metrics),
        "y_true": list(result.y_true),
        "y_pred": list(result.y_pred),
        "validation_size": result.validation_size,
        "best_objective": result.best_objective,
        "n_rules": result.n_rules,
        "best_position_sha256": result.best_position_sha256,
        "selected_config": _json_safe(result.selected_config),
        "selection_failures": _json_safe(result.selection_failures),
        "failure": None if result.failure is None else asdict(result.failure),
    }


def _result_from_dict(data: dict[str, Any]) -> FinalRunResult:
    if data.get("schema_version") != RESULT_SCHEMA:
        raise FinalExecutionError("Final result schema mismatch.")
    task = fr.FinalRunTask(**data["task"])
    metrics_data = data.get("metrics")
    metrics = None if metrics_data is None else FinalMetrics(**metrics_data)
    failure_data = data.get("failure")
    failure = None if failure_data is None else FinalRunFailure(**failure_data)
    result = FinalRunResult(
        task=task,
        status=str(data["status"]),  # type: ignore[arg-type]
        metrics=metrics,
        y_true=tuple(float(v) for v in data.get("y_true", ())),
        y_pred=tuple(float(v) for v in data.get("y_pred", ())),
        validation_size=None if data.get("validation_size") is None else int(data["validation_size"]),
        best_objective=None if data.get("best_objective") is None else float(data["best_objective"]),
        n_rules=None if data.get("n_rules") is None else int(data["n_rules"]),
        best_position_sha256=data.get("best_position_sha256"),
        selected_config=dict(data.get("selected_config", {})),
        selection_failures=tuple(dict(v) for v in data.get("selection_failures", ())),
        failure=failure,
    )
    validate_final_result(result)
    return result


def result_path(workspace: str | os.PathLike[str], run_id: str) -> Path:
    return Path(workspace) / "results" / f"{run_id}.json"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    try:
        with open(tmp, "xb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_final_result(
    workspace: str | os.PathLike[str],
    result: FinalRunResult,
) -> Path:
    validate_final_result(result)
    path = result_path(workspace, result.task.run_id)
    if path.exists():
        raise FinalExecutionError(f"Final result already exists: {path.name}")
    payload = (
        json.dumps(
            _result_dict(result),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(path, payload)
    return path


def load_final_result(path: str | os.PathLike[str]) -> FinalRunResult:
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalExecutionError(f"Invalid Final result file: {p}") from exc
    return _result_from_dict(data)


def load_frozen_manifest(
    workspace: str | os.PathLike[str] = r"outputs\v2\final_external",
) -> fr.FinalManifest:
    return fr.load_final_manifest(
        workspace,
        expected_protocol_fingerprint=FROZEN_PROTOCOL_FINGERPRINT,
        expected_code_commit=FROZEN_MANIFEST_CODE_COMMIT,
    )


def completed_final_results(
    workspace: str | os.PathLike[str],
    manifest: fr.FinalManifest,
) -> tuple[FinalRunResult, ...]:
    results = []
    for task in manifest.tasks:
        path = result_path(workspace, task.run_id)
        if path.is_file():
            result = load_final_result(path)
            if result.task != task:
                raise FinalExecutionError("Stored Final result task differs from frozen manifest task.")
            results.append(result)
    return tuple(results)


def pending_final_tasks(
    workspace: str | os.PathLike[str],
    manifest: fr.FinalManifest,
) -> tuple[fr.FinalRunTask, ...]:
    return tuple(
        task for task in manifest.tasks
        if not result_path(workspace, task.run_id).exists()
    )


__all__ = [
    "FINAL_MANIFEST_FREEZE_TAG",
    "FINAL_MANIFEST_FREEZE_COMMIT",
    "FROZEN_MANIFEST_CODE_COMMIT",
    "FROZEN_PROTOCOL_FINGERPRINT",
    "FROZEN_MANIFEST_SHA256",
    "FROZEN_TOTAL_TASKS",
    "RESULT_SCHEMA",
    "FinalExecutionError",
    "FinalMetrics",
    "FinalRunFailure",
    "FinalRunResult",
    "FinalExecutionPreflight",
    "verify_final_execution_preflight",
    "compute_final_metrics",
    "load_frozen_real_task_series",
    "load_final_task_series",
    "execute_final_task",
    "validate_final_result",
    "result_path",
    "write_final_result",
    "load_final_result",
    "load_frozen_manifest",
    "completed_final_results",
    "pending_final_tasks",
]
