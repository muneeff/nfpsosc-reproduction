from __future__ import annotations

"""Frozen execution contract for the V2 internal ablation study.

This module executes only the 5,250 *new* ablation tasks. The 1,410 PC_NFPSO
reference rows remain cryptographically bound to the already-closed Final
external corpus through the frozen ablation manifest and are never rerun here.
"""

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from . import ablation_runner as ar
from . import final_execution as fe
from . import final_runner as fr
from .ablation import (
    AblationFit,
    fit_ablation_variant_v2,
    forecast_ablation_v2,
)
from .development_runner import pc_nfpso_validation_size


FROZEN_MANIFEST_DIR = Path("configs/v2/ablation_manifest_2026-08-14")
FROZEN_MANIFEST_CODE_COMMIT = "371a533f201797628106cff7845f1bf3b36f1df0"
FROZEN_MANIFEST_FREEZE_COMMIT = "078553e03268340af1bef5e2add47025be5de42d"
FROZEN_MANIFEST_FREEZE_TAG = "v2-ablation-manifest-freeze-2026-08-14"
FROZEN_PROTOCOL_FINGERPRINT = (
    "890c26e2b306e544a67e55e24f4f76d18bc021eff0ded8d8d8e0610ad851d6dc"
)
FROZEN_MANIFEST_SHA256 = (
    "6dedd4dd75c0953d617679226800daec5e072fb45f69af0ed3667f94a6ff376b"
)
FROZEN_SIDECAR_SHA256 = (
    "5976a6bc51c819e1f7acc5db15899c15b29b03a4654e8b607ff6231482a9b8e7"
)
FROZEN_EXTERNAL_FINAL_AGGREGATE = (
    "393b7df1fca6d6dc2e57b60b06d6ccca7f15217e3a8c351b9427346829da299b"
)
FROZEN_TOTAL_TASKS = 5_250
RESULT_SCHEMA = "v2-internal-ablation-result-1"


class AblationExecutionError(RuntimeError):
    """Raised when ablation execution violates the frozen contract."""


@dataclass(frozen=True)
class AblationExecutionPreflight:
    execution_code_commit: str
    manifest_code_commit: str
    manifest_freeze_commit: str
    protocol_fingerprint: str
    manifest_sha256: str
    external_final_results_aggregate_sha256: str
    task_count: int
    selected_radius: float
    selected_alpha: float


@dataclass(frozen=True)
class AblationRunFailure:
    stage: Literal["load", "fit", "forecast", "metrics"]
    error_type: str
    message: str


@dataclass(frozen=True)
class AblationRunResult:
    task: ar.AblationRunTask
    status: Literal["success", "failed"]
    metrics: fe.FinalMetrics | None
    y_true: tuple[float, ...]
    y_pred: tuple[float, ...]
    validation_size: int | None
    best_objective: float | None
    n_rules: int | None
    best_position_sha256: str | None
    selected_config: dict[str, Any]
    failure: AblationRunFailure | None

    @property
    def successful(self) -> bool:
        return self.status == "success"


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise AblationExecutionError(f"Git command failed: {message}") from exc


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_clean_worktree(root: Path) -> None:
    if _git(root, "status", "--porcelain"):
        raise AblationExecutionError(
            "Ablation execution requires a clean Git worktree."
        )


def _assert_manifest_freeze_tag_and_ancestry(root: Path) -> str:
    try:
        tag_target = _git(root, "rev-list", "-n", "1", FROZEN_MANIFEST_FREEZE_TAG)
    except AblationExecutionError as exc:
        raise AblationExecutionError(
            f"Required ablation manifest-freeze tag is missing: "
            f"{FROZEN_MANIFEST_FREEZE_TAG}"
        ) from exc

    if tag_target != FROZEN_MANIFEST_FREEZE_COMMIT:
        raise AblationExecutionError(
            "Ablation manifest-freeze tag target drift detected."
        )

    head = _git(root, "rev-parse", "HEAD")
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FROZEN_MANIFEST_FREEZE_COMMIT, head],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AblationExecutionError(
            "Current execution code is not descended from the frozen ablation manifest."
        )
    return head


def _frozen_manifest_dir(root: Path) -> Path:
    return root / FROZEN_MANIFEST_DIR


def load_frozen_ablation_manifest(
    *, project_root: str | os.PathLike[str] = "."
) -> ar.AblationManifest:
    root = Path(project_root).resolve()
    out = _frozen_manifest_dir(root)
    manifest_path = out / ar.MANIFEST_FILENAME
    sidecar_path = out / ar.MANIFEST_SHA_FILENAME

    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise AblationExecutionError(
            "Frozen ablation manifest or SHA sidecar is missing."
        )
    if _sha256_file(manifest_path) != FROZEN_MANIFEST_SHA256:
        raise AblationExecutionError("Frozen ablation manifest bytes drift detected.")
    if _sha256_file(sidecar_path) != FROZEN_SIDECAR_SHA256:
        raise AblationExecutionError("Frozen ablation manifest sidecar drift detected.")

    try:
        manifest = ar.load_ablation_manifest(
            out,
            expected_protocol_fingerprint=FROZEN_PROTOCOL_FINGERPRINT,
            expected_code_commit=FROZEN_MANIFEST_CODE_COMMIT,
        )
    except Exception as exc:
        raise AblationExecutionError(
            f"Frozen ablation manifest validation failed: {exc}"
        ) from exc

    if manifest.new_task_count != FROZEN_TOTAL_TASKS:
        raise AblationExecutionError("Frozen ablation task count drift detected.")
    if (
        manifest.external_final_results_aggregate_sha256
        != FROZEN_EXTERNAL_FINAL_AGGREGATE
    ):
        raise AblationExecutionError(
            "Frozen external Final reference aggregate drift detected."
        )
    return manifest


def verify_ablation_execution_preflight(
    *, project_root: str | os.PathLike[str] = "."
) -> AblationExecutionPreflight:
    root = Path(project_root).resolve()
    _assert_clean_worktree(root)
    execution_head = _assert_manifest_freeze_tag_and_ancestry(root)

    if ar.protocol_fingerprint(root) != FROZEN_PROTOCOL_FINGERPRINT:
        raise AblationExecutionError("Frozen ablation protocol fingerprint drift detected.")

    # Reuse the already-tested Final environment and real-source integrity gates.
    fr.verify_final_software_environment()
    fe._verify_real_source_snapshot(root)

    manifest = load_frozen_ablation_manifest(project_root=root)
    return AblationExecutionPreflight(
        execution_code_commit=execution_head,
        manifest_code_commit=manifest.code_commit,
        manifest_freeze_commit=FROZEN_MANIFEST_FREEZE_COMMIT,
        protocol_fingerprint=manifest.protocol_fingerprint,
        manifest_sha256=FROZEN_MANIFEST_SHA256,
        external_final_results_aggregate_sha256=(
            manifest.external_final_results_aggregate_sha256
        ),
        task_count=manifest.new_task_count,
        selected_radius=manifest.selected_radius,
        selected_alpha=manifest.selected_alpha,
    )


def _frozen_task_map(
    project_root_resolved: str,
) -> dict[str, ar.AblationRunTask]:
    manifest = load_frozen_ablation_manifest(project_root=project_root_resolved)
    mapping = {task.run_id: task for task in manifest.tasks}
    if len(mapping) != FROZEN_TOTAL_TASKS:
        raise AblationExecutionError("Frozen ablation task map count mismatch.")
    return mapping


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AblationExecutionError("Non-finite float cannot be serialized.")
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


def _best_position_sha256(fitted: AblationFit) -> str:
    if not hasattr(fitted, "optimizer_result"):
        raise AblationExecutionError(
            "Deterministic NF_BASE has no optimizer best position."
        )
    arr = np.asarray(
        fitted.optimizer_result.best_position, dtype="<f8"
    ).reshape(-1)
    if arr.size < 1 or not np.all(np.isfinite(arr)):
        raise AblationExecutionError("Ablation best position is invalid.")
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def _failed(
    task: ar.AblationRunTask,
    stage: str,
    exc: Exception,
    *,
    y_true: Sequence[float] = (),
    y_pred: Sequence[float] = (),
    validation_size: int | None = None,
    selected_config: Mapping[str, Any] | None = None,
) -> AblationRunResult:
    return AblationRunResult(
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
        failure=AblationRunFailure(
            stage=stage,  # type: ignore[arg-type]
            error_type=type(exc).__name__,
            message=str(exc),
        ),
    )


def execute_ablation_task(
    task: ar.AblationRunTask,
    *,
    project_root: str | os.PathLike[str] = ".",
    fit_function: Callable[..., AblationFit] = fit_ablation_variant_v2,
    forecast_function: Callable[[AblationFit, np.ndarray], np.ndarray] = forecast_ablation_v2,
    series_loader: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
) -> AblationRunResult:
    root = str(Path(project_root).resolve())
    frozen = _frozen_task_map(root).get(task.run_id)
    if frozen != task:
        raise AblationExecutionError(
            "Task is not an exact member of the frozen ablation manifest."
        )

    loader = fe.load_final_task_series if series_loader is None else series_loader
    try:
        train, test = loader(task, project_root=project_root)
        train = np.asarray(train, dtype=float).reshape(-1)
        test = np.asarray(test, dtype=float).reshape(-1)
        if (
            train.size < 1
            or test.size < 1
            or not np.all(np.isfinite(train))
            or not np.all(np.isfinite(test))
        ):
            raise AblationExecutionError("Ablation train/test arrays are invalid.")
    except Exception as exc:
        return _failed(task, "load", exc)

    validation_size = pc_nfpso_validation_size(len(train), task.n_lags)
    try:
        fitted = fit_function(
            task.variant,
            train,
            n_lags=task.n_lags,
            validation_size=validation_size,
            radius=task.radius,
            alpha=task.alpha,
            optimizer_seed=task.optimizer_seed,
        )
    except Exception as exc:
        return _failed(
            task,
            "fit",
            exc,
            y_true=test,
            validation_size=validation_size,
        )

    try:
        predicted = np.asarray(
            forecast_function(fitted, test), dtype=float
        ).reshape(-1)
        if predicted.shape != test.shape or not np.all(np.isfinite(predicted)):
            raise AblationExecutionError("Ablation forecast is invalid.")
    except Exception as exc:
        return _failed(
            task,
            "forecast",
            exc,
            y_true=test,
            validation_size=validation_size,
        )

    try:
        metrics = fe.compute_final_metrics(
            train,
            test,
            predicted,
            seasonal_period=task.seasonal_period,
        )
        n_rules = int(fitted.final_model.n_rules)
        if n_rules < 1:
            raise AblationExecutionError("Ablation fit produced no fuzzy rules.")

        if task.variant == "NF_BASE":
            best_objective = None
            best_position_hash = None
            selected_config = {
                "variant": task.variant,
                "pso": False,
                "dynamics": None,
                "boundary": None,
            }
        else:
            best_objective = float(fitted.optimizer_result.best_cost)
            if not np.isfinite(best_objective):
                raise AblationExecutionError("Ablation best objective is invalid.")
            best_position_hash = _best_position_sha256(fitted)
            selected_config = {
                "variant": task.variant,
                "pso": True,
                "dynamics": str(fitted.dynamics),
                "boundary": str(fitted.boundary),
                "objective_components": _json_safe(
                    fitted.selected_candidate.components
                ),
            }
    except Exception as exc:
        return _failed(
            task,
            "metrics",
            exc,
            y_true=test,
            y_pred=predicted,
            validation_size=validation_size,
        )

    result = AblationRunResult(
        task=task,
        status="success",
        metrics=metrics,
        y_true=tuple(float(v) for v in test),
        y_pred=tuple(float(v) for v in predicted),
        validation_size=validation_size,
        best_objective=best_objective,
        n_rules=n_rules,
        best_position_sha256=best_position_hash,
        selected_config=dict(selected_config),
        failure=None,
    )
    validate_ablation_result(result, project_root=project_root)
    return result


def validate_ablation_result(
    result: AblationRunResult,
    *,
    project_root: str | os.PathLike[str] = ".",
) -> None:
    task = result.task
    frozen = _frozen_task_map(str(Path(project_root).resolve())).get(task.run_id)
    if frozen != task:
        raise AblationExecutionError(
            "Result task is not in the frozen ablation design."
        )

    if result.status == "success":
        if result.failure is not None or result.metrics is None:
            raise AblationExecutionError(
                "Successful ablation result has inconsistent status/provenance."
            )
        y = np.asarray(result.y_true, dtype=float)
        p = np.asarray(result.y_pred, dtype=float)
        if (
            y.shape != p.shape
            or y.size < 1
            or not np.all(np.isfinite(y))
            or not np.all(np.isfinite(p))
        ):
            raise AblationExecutionError(
                "Successful ablation result contains invalid prediction arrays."
            )
        for value in asdict(result.metrics).values():
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise AblationExecutionError(
                    "Successful ablation result contains an invalid metric."
                )
        if result.validation_size is None or result.n_rules is None or result.n_rules < 1:
            raise AblationExecutionError(
                "Successful ablation result lacks fit provenance."
            )

        if task.variant == "NF_BASE":
            if result.best_objective is not None or result.best_position_sha256 is not None:
                raise AblationExecutionError(
                    "NF_BASE result illegally contains PSO provenance."
                )
        else:
            if result.best_objective is None or not np.isfinite(result.best_objective):
                raise AblationExecutionError(
                    "Stochastic ablation result lacks a valid best objective."
                )
            if (
                result.best_position_sha256 is None
                or len(result.best_position_sha256) != 64
            ):
                raise AblationExecutionError(
                    "Stochastic ablation result lacks best-position hash."
                )
    elif result.status == "failed":
        if result.failure is None or result.metrics is not None:
            raise AblationExecutionError(
                "Failed ablation result has inconsistent failure provenance."
            )
    else:
        raise AblationExecutionError(
            f"Invalid ablation result status: {result.status!r}"
        )


def _result_dict(result: AblationRunResult) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA,
        "task": asdict(result.task),
        "status": result.status,
        "metrics": None if result.metrics is None else asdict(result.metrics),
        "y_true": list(result.y_true),
        "y_pred": list(result.y_pred),
        "validation_size": result.validation_size,
        "best_objective": result.best_objective,
        "n_rules": result.n_rules,
        "best_position_sha256": result.best_position_sha256,
        "selected_config": _json_safe(result.selected_config),
        "failure": None if result.failure is None else asdict(result.failure),
    }


def _result_from_dict(
    data: dict[str, Any],
    *,
    project_root: str | os.PathLike[str] = ".",
) -> AblationRunResult:
    if data.get("schema_version") != RESULT_SCHEMA:
        raise AblationExecutionError("Ablation result schema mismatch.")

    task = ar.AblationRunTask(**data["task"])
    metrics_data = data.get("metrics")
    metrics = None if metrics_data is None else fe.FinalMetrics(**metrics_data)
    failure_data = data.get("failure")
    failure = (
        None
        if failure_data is None
        else AblationRunFailure(**failure_data)
    )
    result = AblationRunResult(
        task=task,
        status=str(data["status"]),  # type: ignore[arg-type]
        metrics=metrics,
        y_true=tuple(float(v) for v in data.get("y_true", ())),
        y_pred=tuple(float(v) for v in data.get("y_pred", ())),
        validation_size=(
            None
            if data.get("validation_size") is None
            else int(data["validation_size"])
        ),
        best_objective=(
            None
            if data.get("best_objective") is None
            else float(data["best_objective"])
        ),
        n_rules=None if data.get("n_rules") is None else int(data["n_rules"]),
        best_position_sha256=data.get("best_position_sha256"),
        selected_config=dict(data.get("selected_config", {})),
        failure=failure,
    )
    validate_ablation_result(result, project_root=project_root)
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


def write_ablation_result(
    workspace: str | os.PathLike[str],
    result: AblationRunResult,
    *,
    project_root: str | os.PathLike[str] = ".",
) -> Path:
    validate_ablation_result(result, project_root=project_root)
    path = result_path(workspace, result.task.run_id)
    if path.exists():
        raise AblationExecutionError(
            f"Ablation result already exists: {path.name}"
        )
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


def load_ablation_result(
    path: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str] = ".",
) -> AblationRunResult:
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AblationExecutionError(
            f"Invalid ablation result file: {p}"
        ) from exc
    return _result_from_dict(data, project_root=project_root)


def completed_ablation_results(
    workspace: str | os.PathLike[str],
    manifest: ar.AblationManifest,
    *,
    project_root: str | os.PathLike[str] = ".",
) -> tuple[AblationRunResult, ...]:
    results = []
    for task in manifest.tasks:
        path = result_path(workspace, task.run_id)
        if path.is_file():
            result = load_ablation_result(path, project_root=project_root)
            if result.task != task:
                raise AblationExecutionError(
                    "Stored ablation result task differs from frozen manifest task."
                )
            results.append(result)
    return tuple(results)


def pending_ablation_tasks(
    workspace: str | os.PathLike[str],
    manifest: ar.AblationManifest,
) -> tuple[ar.AblationRunTask, ...]:
    return tuple(
        task
        for task in manifest.tasks
        if not result_path(workspace, task.run_id).exists()
    )


__all__ = [
    "FROZEN_MANIFEST_CODE_COMMIT",
    "FROZEN_MANIFEST_FREEZE_COMMIT",
    "FROZEN_MANIFEST_FREEZE_TAG",
    "FROZEN_PROTOCOL_FINGERPRINT",
    "FROZEN_MANIFEST_SHA256",
    "FROZEN_SIDECAR_SHA256",
    "FROZEN_EXTERNAL_FINAL_AGGREGATE",
    "FROZEN_TOTAL_TASKS",
    "RESULT_SCHEMA",
    "AblationExecutionError",
    "AblationExecutionPreflight",
    "AblationRunFailure",
    "AblationRunResult",
    "load_frozen_ablation_manifest",
    "verify_ablation_execution_preflight",
    "execute_ablation_task",
    "validate_ablation_result",
    "result_path",
    "write_ablation_result",
    "load_ablation_result",
    "completed_ablation_results",
    "pending_ablation_tasks",
]
