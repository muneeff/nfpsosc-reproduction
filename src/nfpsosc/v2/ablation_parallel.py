from __future__ import annotations

"""Fail-closed parallel scheduler for the frozen V2 internal ablation study."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

from . import ablation_execution as ae
from . import ablation_runner as ar


DEFAULT_WORKERS = 1
DEFAULT_IN_FLIGHT_MULTIPLIER = 2
THREADS_PER_WORKER = 1
THREAD_ENV_NAMES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

EXECUTION_FREEZE_SCHEMA = "v2-internal-ablation-execution-workspace-freeze-1"
EXECUTION_FREEZE_STATUS = "LOCKED_PRE_ABLATION_OUTCOMES"
EXECUTION_FREEZE_FILENAME = "ablation_execution_workspace_freeze.json"
EXECUTION_FREEZE_SHA_FILENAME = "ablation_execution_workspace_freeze.sha256"
ABLATION_EXECUTION_FREEZE_TAG = "v2-ablation-execution-freeze-2026-08-14"


class AblationParallelError(RuntimeError):
    """Raised when ablation scheduling violates the frozen execution contract."""


@dataclass(frozen=True)
class AblationWorkspaceState:
    execution_code_commit: str
    manifest_code_commit: str
    manifest_freeze_commit: str
    protocol_fingerprint: str
    manifest_sha256: str
    task_count: int
    completed_count: int
    successful_count: int
    failed_count: int
    pending_count: int


@dataclass(frozen=True)
class AblationParallelSummary:
    workers: int
    max_in_flight: int
    selected_count: int
    completed_this_call: int
    successful_this_call: int
    failed_this_call: int
    completed_total: int
    successful_total: int
    failed_total: int
    pending_total: int
    wall_seconds: float
    stopped_after_failure: bool


def _validate_workers(workers: int) -> int:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise AblationParallelError("workers must be a positive integer.")
    return int(workers)


def _validate_max_runs(max_runs: int | None) -> int | None:
    if max_runs is None:
        return None
    if isinstance(max_runs, bool) or not isinstance(max_runs, int) or max_runs < 0:
        raise AblationParallelError(
            "max_runs must be a non-negative integer or None."
        )
    return int(max_runs)


def _configure_worker_threads() -> None:
    value = str(THREADS_PER_WORKER)
    for name in THREAD_ENV_NAMES:
        os.environ[name] = value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
        raise AblationParallelError(f"Git command failed: {message}") from exc


def _task_payload(
    task: ar.AblationRunTask,
    project_root: str | os.PathLike[str],
) -> dict[str, object]:
    return {
        "project_root": str(Path(project_root).resolve()),
        "task": {
            "run_id": str(task.run_id),
            "domain": str(task.domain),
            "variant": str(task.variant),
            "generator": task.generator,
            "dgp_seed": task.dgp_seed,
            "noise_level": task.noise_level,
            "source": task.source,
            "frequency": task.frequency,
            "series_id": task.series_id,
            "optimizer_seed": task.optimizer_seed,
            "seasonal_period": int(task.seasonal_period),
            "n_lags": int(task.n_lags),
            "radius": float(task.radius),
            "alpha": float(task.alpha),
        },
    }


def _execute_task_payload(payload: Mapping[str, object]):
    from .ablation_execution import execute_ablation_task
    from .ablation_runner import AblationRunTask

    raw = payload["task"]
    if not isinstance(raw, Mapping):
        raise AblationParallelError("Worker task payload is invalid.")

    task = AblationRunTask(
        run_id=str(raw["run_id"]),
        domain=str(raw["domain"]),  # type: ignore[arg-type]
        variant=str(raw["variant"]),
        generator=None if raw["generator"] is None else str(raw["generator"]),
        dgp_seed=None if raw["dgp_seed"] is None else int(raw["dgp_seed"]),
        noise_level=(
            None if raw["noise_level"] is None else float(raw["noise_level"])
        ),
        source=None if raw["source"] is None else str(raw["source"]),
        frequency=None if raw["frequency"] is None else str(raw["frequency"]),
        series_id=None if raw["series_id"] is None else str(raw["series_id"]),
        optimizer_seed=(
            None if raw["optimizer_seed"] is None else int(raw["optimizer_seed"])
        ),
        seasonal_period=int(raw["seasonal_period"]),
        n_lags=int(raw["n_lags"]),
        radius=float(raw["radius"]),
        alpha=float(raw["alpha"]),
    )
    return execute_ablation_task(
        task,
        project_root=str(payload["project_root"]),
    )


def _production_executor(max_workers: int):
    context = mp.get_context("spawn")
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
        initializer=_configure_worker_threads,
    )


def _execution_freeze_payload(
    preflight: ae.AblationExecutionPreflight,
    *,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, object]:
    workers = _validate_workers(workers)
    max_in_flight = max(workers, workers * DEFAULT_IN_FLIGHT_MULTIPLIER)
    return {
        "schema_version": EXECUTION_FREEZE_SCHEMA,
        "created_date": "2026-08-14",
        "status": EXECUTION_FREEZE_STATUS,
        "execution_code_commit": str(preflight.execution_code_commit),
        "manifest_code_commit": str(preflight.manifest_code_commit),
        "manifest_freeze_commit": str(preflight.manifest_freeze_commit),
        "protocol_fingerprint": str(preflight.protocol_fingerprint),
        "manifest_sha256": str(preflight.manifest_sha256),
        "external_final_results_aggregate_sha256": str(
            preflight.external_final_results_aggregate_sha256
        ),
        "task_count": int(preflight.task_count),
        "selected_radius": float(preflight.selected_radius),
        "selected_alpha": float(preflight.selected_alpha),
        "parallel_policy": {
            "default_workers": workers,
            "max_in_flight": max_in_flight,
            "in_flight_multiplier": DEFAULT_IN_FLIGHT_MULTIPLIER,
            "threads_per_worker": THREADS_PER_WORKER,
            "thread_environment_variables": list(THREAD_ENV_NAMES),
            "multiprocessing_start_method": "spawn",
            "parent_only_checkpoint_writes": True,
            "fail_fast_stop_new_submissions": True,
            "drain_already_running_after_scientific_failure": True,
            "resume_from_immutable_per_run_checkpoints": True,
            "frozen_worker_count_enforced": True,
        },
        "result_state_at_freeze": {
            "completed": 0,
            "successful": 0,
            "failed": 0,
            "pending": int(preflight.task_count),
            "ablation_outcomes_generated": False,
        },
        "execution_gate": {
            "required_tag": ABLATION_EXECUTION_FREEZE_TAG,
            "required_tag_target": str(preflight.execution_code_commit),
            "tag_required_before_any_task_execution": True,
        },
        "lock_rule": (
            "After this workspace freeze is created, ablation execution is permitted "
            "only from this exact execution_code_commit, with the frozen worker count, "
            "and only after the required annotated execution-freeze tag points to that "
            "commit. No code/protocol/manifest change is admissible in response to "
            "observed ablation outcomes."
        ),
    }


def write_execution_workspace_freeze(
    workspace: str | os.PathLike[str],
    payload: Mapping[str, object],
) -> tuple[Path, Path]:
    out = Path(workspace)
    freeze_path = out / EXECUTION_FREEZE_FILENAME
    sha_path = out / EXECUTION_FREEZE_SHA_FILENAME
    if freeze_path.exists() or sha_path.exists():
        raise AblationParallelError(
            "Ablation execution workspace freeze already exists."
        )

    encoded = (
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    digest = _sha256_bytes(encoded)
    _atomic_write(freeze_path, encoded)
    try:
        _atomic_write(sha_path, (digest + "\n").encode("ascii"))
    except Exception:
        freeze_path.unlink(missing_ok=True)
        raise
    return freeze_path, sha_path


def load_execution_workspace_freeze(
    workspace: str | os.PathLike[str],
) -> dict[str, object]:
    out = Path(workspace)
    freeze_path = out / EXECUTION_FREEZE_FILENAME
    sha_path = out / EXECUTION_FREEZE_SHA_FILENAME
    if not freeze_path.is_file() or not sha_path.is_file():
        raise AblationParallelError(
            "Ablation execution workspace freeze is missing."
        )

    payload = freeze_path.read_bytes()
    expected = sha_path.read_text(encoding="ascii").strip()
    actual = _sha256_bytes(payload)
    if actual != expected:
        raise AblationParallelError(
            "Ablation execution workspace freeze SHA-256 mismatch."
        )
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AblationParallelError(
            "Ablation execution workspace freeze is invalid JSON."
        ) from exc

    if data.get("schema_version") != EXECUTION_FREEZE_SCHEMA:
        raise AblationParallelError("Ablation execution freeze schema mismatch.")
    if data.get("status") != EXECUTION_FREEZE_STATUS:
        raise AblationParallelError("Ablation execution freeze is not locked.")

    state = data.get("result_state_at_freeze", {})
    if not isinstance(state, Mapping):
        raise AblationParallelError("Ablation freeze result state is invalid.")
    if (
        int(state.get("completed", -1)) != 0
        or int(state.get("successful", -1)) != 0
        or int(state.get("failed", -1)) != 0
        or bool(state.get("ablation_outcomes_generated", True))
    ):
        raise AblationParallelError(
            "Execution workspace was not frozen before ablation outcomes."
        )
    return data


def _result_files(workspace: str | os.PathLike[str]) -> tuple[Path, ...]:
    results_dir = Path(workspace) / "results"
    if not results_dir.exists():
        return ()
    if not results_dir.is_dir():
        raise AblationParallelError(
            "Ablation results path exists but is not a directory."
        )

    unexpected = [
        p
        for p in results_dir.iterdir()
        if not (p.is_file() and p.suffix == ".json" and len(p.stem) == 64)
    ]
    if unexpected:
        names = ", ".join(sorted(p.name for p in unexpected)[:10])
        raise AblationParallelError(
            f"Unexpected entries in ablation results directory: {names}"
        )
    return tuple(sorted(results_dir.glob("*.json")))


def inspect_ablation_workspace_results(
    workspace: str | os.PathLike[str],
    manifest: ar.AblationManifest,
    *,
    project_root: str | os.PathLike[str] = ".",
) -> tuple[int, int, int, int]:
    task_by_id = {task.run_id: task for task in manifest.tasks}
    if len(task_by_id) != manifest.new_task_count:
        raise AblationParallelError(
            "Frozen ablation manifest contains duplicate run IDs."
        )

    completed = successful = failed = 0
    for path in _result_files(workspace):
        if path.stem not in task_by_id:
            raise AblationParallelError(
                f"Result file is not part of frozen ablation manifest: {path.name}"
            )
        result = ae.load_ablation_result(path, project_root=project_root)
        if result.task != task_by_id[path.stem]:
            raise AblationParallelError(
                f"Stored ablation result task mismatch: {path.name}"
            )
        completed += 1
        if result.successful:
            successful += 1
        else:
            failed += 1

    pending = int(manifest.new_task_count) - completed
    if pending < 0:
        raise AblationParallelError(
            "Ablation workspace has more results than frozen tasks."
        )
    return completed, successful, failed, pending


def prepare_ablation_execution_workspace(
    workspace: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str] = ".",
    workers: int = DEFAULT_WORKERS,
) -> tuple[AblationWorkspaceState, bool]:
    workers = _validate_workers(workers)
    preflight = ae.verify_ablation_execution_preflight(project_root=project_root)
    manifest = ae.load_frozen_ablation_manifest(project_root=project_root)

    completed, successful, failed, pending = inspect_ablation_workspace_results(
        workspace,
        manifest,
        project_root=project_root,
    )
    if completed != 0:
        raise AblationParallelError(
            "Cannot create/verify pre-ablation execution workspace with existing results."
        )

    out = Path(workspace)
    freeze_path = out / EXECUTION_FREEZE_FILENAME
    sha_path = out / EXECUTION_FREEZE_SHA_FILENAME
    created = False

    if freeze_path.exists() or sha_path.exists():
        if not (freeze_path.is_file() and sha_path.is_file()):
            raise AblationParallelError(
                "Incomplete ablation execution workspace freeze."
            )
        freeze = load_execution_workspace_freeze(out)
    else:
        freeze = _execution_freeze_payload(preflight, workers=workers)
        write_execution_workspace_freeze(out, freeze)
        freeze = load_execution_workspace_freeze(out)
        created = True

    if str(freeze["execution_code_commit"]) != preflight.execution_code_commit:
        raise AblationParallelError(
            "Current Git HEAD differs from frozen ablation execution code commit."
        )
    if str(freeze["manifest_code_commit"]) != preflight.manifest_code_commit:
        raise AblationParallelError("Frozen manifest-code commit mismatch.")
    if str(freeze["manifest_freeze_commit"]) != preflight.manifest_freeze_commit:
        raise AblationParallelError("Frozen manifest-freeze commit mismatch.")
    if str(freeze["protocol_fingerprint"]) != preflight.protocol_fingerprint:
        raise AblationParallelError("Frozen protocol fingerprint mismatch.")
    if str(freeze["manifest_sha256"]) != preflight.manifest_sha256:
        raise AblationParallelError("Frozen manifest SHA-256 mismatch.")
    if int(freeze["task_count"]) != preflight.task_count:
        raise AblationParallelError("Frozen task count mismatch.")
    if float(freeze["selected_radius"]) != preflight.selected_radius:
        raise AblationParallelError("Frozen selected radius mismatch.")
    if float(freeze["selected_alpha"]) != preflight.selected_alpha:
        raise AblationParallelError("Frozen selected alpha mismatch.")

    policy = freeze.get("parallel_policy", {})
    if not isinstance(policy, Mapping):
        raise AblationParallelError("Frozen parallel policy is invalid.")
    if int(policy.get("default_workers", -1)) != workers:
        raise AblationParallelError(
            "Requested worker count differs from frozen ablation worker count."
        )

    return (
        AblationWorkspaceState(
            execution_code_commit=preflight.execution_code_commit,
            manifest_code_commit=preflight.manifest_code_commit,
            manifest_freeze_commit=preflight.manifest_freeze_commit,
            protocol_fingerprint=preflight.protocol_fingerprint,
            manifest_sha256=preflight.manifest_sha256,
            task_count=preflight.task_count,
            completed_count=completed,
            successful_count=successful,
            failed_count=failed,
            pending_count=pending,
        ),
        created,
    )


def verify_ablation_execution_workspace(
    workspace: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str] = ".",
    workers: int | None = None,
) -> AblationWorkspaceState:
    preflight = ae.verify_ablation_execution_preflight(project_root=project_root)
    freeze = load_execution_workspace_freeze(workspace)

    if str(freeze["execution_code_commit"]) != preflight.execution_code_commit:
        raise AblationParallelError(
            "Current Git HEAD differs from frozen ablation execution code commit."
        )
    if str(freeze["manifest_code_commit"]) != preflight.manifest_code_commit:
        raise AblationParallelError("Frozen manifest-code commit mismatch.")
    if str(freeze["manifest_freeze_commit"]) != preflight.manifest_freeze_commit:
        raise AblationParallelError("Frozen manifest-freeze commit mismatch.")
    if str(freeze["protocol_fingerprint"]) != preflight.protocol_fingerprint:
        raise AblationParallelError("Frozen protocol fingerprint mismatch.")
    if str(freeze["manifest_sha256"]) != preflight.manifest_sha256:
        raise AblationParallelError("Frozen manifest SHA-256 mismatch.")
    if int(freeze["task_count"]) != preflight.task_count:
        raise AblationParallelError("Frozen task count mismatch.")

    policy = freeze.get("parallel_policy", {})
    if not isinstance(policy, Mapping):
        raise AblationParallelError("Frozen parallel policy is invalid.")
    frozen_workers = int(policy.get("default_workers", -1))
    if workers is not None and _validate_workers(workers) != frozen_workers:
        raise AblationParallelError(
            "Requested worker count differs from frozen ablation worker count."
        )

    manifest = ae.load_frozen_ablation_manifest(project_root=project_root)
    completed, successful, failed, pending = inspect_ablation_workspace_results(
        workspace,
        manifest,
        project_root=project_root,
    )
    return AblationWorkspaceState(
        execution_code_commit=preflight.execution_code_commit,
        manifest_code_commit=preflight.manifest_code_commit,
        manifest_freeze_commit=preflight.manifest_freeze_commit,
        protocol_fingerprint=preflight.protocol_fingerprint,
        manifest_sha256=preflight.manifest_sha256,
        task_count=preflight.task_count,
        completed_count=completed,
        successful_count=successful,
        failed_count=failed,
        pending_count=pending,
    )


def assert_execution_freeze_tag(
    workspace: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str],
) -> None:
    root = Path(project_root).resolve()
    freeze = load_execution_workspace_freeze(workspace)
    expected = str(freeze["execution_code_commit"])
    try:
        target = _git(root, "rev-list", "-n", "1", ABLATION_EXECUTION_FREEZE_TAG)
    except AblationParallelError as exc:
        raise AblationParallelError(
            f"Required ablation execution-freeze tag is missing: "
            f"{ABLATION_EXECUTION_FREEZE_TAG}"
        ) from exc
    if target != expected:
        raise AblationParallelError(
            f"Ablation execution-freeze tag target mismatch: "
            f"expected {expected}, got {target}."
        )


def _schedule_selected_tasks(
    selected: Sequence[ar.AblationRunTask],
    *,
    workspace: str | os.PathLike[str],
    project_root: str | os.PathLike[str],
    workers: int,
    fail_fast: bool,
    progress_callback: Callable[[int, int, Any], None] | None,
    executor_factory: Callable[[int], Any],
    worker_function: Callable[[Mapping[str, object]], Any],
) -> tuple[int, int, int, bool, float]:
    selected = tuple(selected)
    if not selected:
        return 0, 0, 0, False, 0.0

    max_in_flight = max(workers, workers * DEFAULT_IN_FLIGHT_MULTIPLIER)
    started = time.perf_counter()
    completed_this_call = successful_this_call = failed_this_call = 0
    stopped_after_failure = False

    task_iter = iter(selected)
    in_flight: dict[Any, ar.AblationRunTask] = {}

    def submit_next(executor: Any) -> bool:
        try:
            task = next(task_iter)
        except StopIteration:
            return False
        future = executor.submit(
            worker_function,
            _task_payload(task, project_root),
        )
        in_flight[future] = task
        return True

    with executor_factory(workers) as executor:
        while len(in_flight) < max_in_flight and submit_next(executor):
            pass

        while in_flight:
            done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
            ordered_done = sorted(
                done,
                key=lambda future: in_flight[future].run_id,
            )

            for future in ordered_done:
                task = in_flight.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    for other in in_flight:
                        other.cancel()
                    raise AblationParallelError(
                        "Worker process raised outside the frozen ablation task-level "
                        f"failure contract for run {task.run_id}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc

                if result.task != task:
                    for other in in_flight:
                        other.cancel()
                    raise AblationParallelError(
                        "Worker returned a result for the wrong frozen ablation task."
                    )

                ae.validate_ablation_result(result, project_root=project_root)
                ae.write_ablation_result(
                    workspace,
                    result,
                    project_root=project_root,
                )
                completed_this_call += 1
                if result.successful:
                    successful_this_call += 1
                else:
                    failed_this_call += 1
                    stopped_after_failure = True

                if progress_callback is not None:
                    progress_callback(
                        completed_this_call,
                        len(selected),
                        result,
                    )

            if not (fail_fast and stopped_after_failure):
                while len(in_flight) < max_in_flight and submit_next(executor):
                    pass

    wall_seconds = time.perf_counter() - started
    return (
        completed_this_call,
        successful_this_call,
        failed_this_call,
        stopped_after_failure,
        float(wall_seconds),
    )


def run_pending_ablation_tasks_parallel(
    workspace: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str] = ".",
    workers: int = DEFAULT_WORKERS,
    max_runs: int | None = None,
    progress_callback: Callable[[int, int, Any], None] | None = None,
    _executor_factory: Callable[[int], Any] | None = None,
    _worker_function: Callable[[Mapping[str, object]], Any] | None = None,
) -> AblationParallelSummary:
    workers = _validate_workers(workers)
    max_runs = _validate_max_runs(max_runs)

    freeze_exists = (
        (Path(workspace) / EXECUTION_FREEZE_FILENAME).exists()
        or (Path(workspace) / EXECUTION_FREEZE_SHA_FILENAME).exists()
    )

    if not freeze_exists:
        if max_runs != 0:
            raise AblationParallelError(
                "Ablation execution workspace has not been frozen. First run "
                "exactly with max_runs=0, inspect/back up the freeze, and create "
                f"the {ABLATION_EXECUTION_FREEZE_TAG} tag before any task."
            )
        before, created = prepare_ablation_execution_workspace(
            workspace,
            project_root=project_root,
            workers=workers,
        )
        if not created:
            raise AblationParallelError(
                "Expected to create the ablation execution workspace freeze."
            )
    else:
        before = verify_ablation_execution_workspace(
            workspace,
            project_root=project_root,
            workers=workers,
        )

    if before.failed_count:
        raise AblationParallelError(
            "Fail-closed gate: ablation workspace already contains failed runs "
            f"({before.failed_count}); refusing additional tasks."
        )

    manifest = ae.load_frozen_ablation_manifest(project_root=project_root)
    pending = ae.pending_ablation_tasks(workspace, manifest)
    if max_runs is not None:
        pending = pending[:max_runs]

    selected_count = len(pending)
    max_in_flight = max(workers, workers * DEFAULT_IN_FLIGHT_MULTIPLIER)

    if selected_count == 0:
        return AblationParallelSummary(
            workers=workers,
            max_in_flight=max_in_flight,
            selected_count=0,
            completed_this_call=0,
            successful_this_call=0,
            failed_this_call=0,
            completed_total=before.completed_count,
            successful_total=before.successful_count,
            failed_total=before.failed_count,
            pending_total=before.pending_count,
            wall_seconds=0.0,
            stopped_after_failure=False,
        )

    assert_execution_freeze_tag(
        workspace,
        project_root=project_root,
    )

    executor_factory = _executor_factory or _production_executor
    worker_function = _worker_function or _execute_task_payload

    (
        completed_this_call,
        successful_this_call,
        failed_this_call,
        stopped_after_failure,
        wall_seconds,
    ) = _schedule_selected_tasks(
        pending,
        workspace=workspace,
        project_root=project_root,
        workers=workers,
        fail_fast=True,
        progress_callback=progress_callback,
        executor_factory=executor_factory,
        worker_function=worker_function,
    )

    after = verify_ablation_execution_workspace(
        workspace,
        project_root=project_root,
        workers=workers,
    )
    expected_completed = before.completed_count + completed_this_call
    if after.completed_count != expected_completed:
        raise AblationParallelError(
            "Post-run ablation workspace completed-count mismatch."
        )
    if (
        after.successful_count
        != before.successful_count + successful_this_call
    ):
        raise AblationParallelError(
            "Post-run ablation successful-count mismatch."
        )
    if after.failed_count != before.failed_count + failed_this_call:
        raise AblationParallelError(
            "Post-run ablation failed-count mismatch."
        )

    return AblationParallelSummary(
        workers=workers,
        max_in_flight=max_in_flight,
        selected_count=selected_count,
        completed_this_call=completed_this_call,
        successful_this_call=successful_this_call,
        failed_this_call=failed_this_call,
        completed_total=after.completed_count,
        successful_total=after.successful_count,
        failed_total=after.failed_count,
        pending_total=after.pending_count,
        wall_seconds=wall_seconds,
        stopped_after_failure=stopped_after_failure,
    )


__all__ = [
    "DEFAULT_WORKERS",
    "DEFAULT_IN_FLIGHT_MULTIPLIER",
    "THREADS_PER_WORKER",
    "THREAD_ENV_NAMES",
    "EXECUTION_FREEZE_SCHEMA",
    "EXECUTION_FREEZE_STATUS",
    "EXECUTION_FREEZE_FILENAME",
    "EXECUTION_FREEZE_SHA_FILENAME",
    "ABLATION_EXECUTION_FREEZE_TAG",
    "AblationParallelError",
    "AblationWorkspaceState",
    "AblationParallelSummary",
    "write_execution_workspace_freeze",
    "load_execution_workspace_freeze",
    "inspect_ablation_workspace_results",
    "prepare_ablation_execution_workspace",
    "verify_ablation_execution_workspace",
    "assert_execution_freeze_tag",
    "run_pending_ablation_tasks_parallel",
]
