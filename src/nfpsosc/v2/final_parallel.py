"""
Bounded parallel scheduler for the frozen V2 Final external benchmark.

This layer changes execution scheduling only. Scientific task identity,
hyperparameters, series identities, fitting/forecasting, metrics, provenance,
and per-run checkpoint semantics remain owned by final_runner.py and
final_execution.py.

Pre-Final execution contract
----------------------------
* the already frozen 8,780-task Final manifest is immutable;
* the manifest/protocol freeze tag must remain valid;
* the execution workspace is frozen at an exact committed execution HEAD before
  the first Final task is allowed to run;
* actual Final execution additionally requires an annotated execution-freeze tag
  targeting that exact HEAD;
* four workers by default, spawn multiprocessing, one BLAS/OpenMP thread each;
* primitive worker payloads so thread caps are installed before scientific
  imports in spawned workers;
* only the parent process writes result checkpoints;
* at most 2 x workers tasks are in flight;
* existing/new scientific failures stop submission of new work, while already
  running tasks are drained and checkpointed;
* resume uses immutable per-run checkpoints; overwrites are forbidden.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence


DEFAULT_WORKERS = 4
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

EXECUTION_FREEZE_SCHEMA = "v2-final-execution-workspace-freeze-1"
EXECUTION_FREEZE_STATUS = "LOCKED_PRE_FINAL_OUTCOMES"
EXECUTION_FREEZE_FILENAME = "final_execution_workspace_freeze.json"
EXECUTION_FREEZE_SHA_FILENAME = "final_execution_workspace_freeze.sha256"
FINAL_EXECUTION_FREEZE_TAG = "v2-final-memory-safe-execution-freeze-2026-08-16"

class FinalParallelError(RuntimeError):
    """Raised when Final parallel execution violates the frozen contract."""


@dataclass(frozen=True)
class FinalWorkspaceState:
    execution_code_commit: str
    manifest_code_commit: str
    protocol_fingerprint: str
    manifest_sha256: str
    task_count: int
    completed_count: int
    successful_count: int
    failed_count: int
    pending_count: int


@dataclass(frozen=True)
class FinalParallelSummary:
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
        raise FinalParallelError("workers must be a positive integer.")
    return int(workers)


def _validate_max_runs(max_runs: int | None) -> int | None:
    if max_runs is None:
        return None
    if isinstance(max_runs, bool) or not isinstance(max_runs, int) or max_runs < 0:
        raise FinalParallelError("max_runs must be a non-negative integer or None.")
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
        raise FinalParallelError(f"Git command failed: {' '.join(args)}") from exc
    return completed.stdout.strip()


def _task_payload(task: Any, project_root: str | os.PathLike[str]) -> dict[str, object]:
    return {
        "project_root": str(Path(project_root).resolve()),
        "task": {
            "run_id": str(task.run_id),
            "domain": str(task.domain),
            "method": str(task.method),
            "generator": task.generator,
            "dgp_seed": task.dgp_seed,
            "noise_level": task.noise_level,
            "source": task.source,
            "frequency": task.frequency,
            "series_id": task.series_id,
            "optimizer_seed": task.optimizer_seed,
            "seasonal_period": int(task.seasonal_period),
            "n_lags": int(task.n_lags),
            "radius": task.radius,
            "alpha": task.alpha,
        },
    }


def _execute_task_payload(payload: Mapping[str, object]):
    # Import only after the spawned worker initializer has installed one-thread
    # BLAS/OpenMP caps.
    from .final_runner import FinalRunTask
    from .final_execution import execute_final_task

    raw = payload["task"]
    if not isinstance(raw, Mapping):
        raise FinalParallelError("Worker task payload is invalid.")
    task = FinalRunTask(
        run_id=str(raw["run_id"]),
        domain=str(raw["domain"]),  # type: ignore[arg-type]
        method=str(raw["method"]),
        generator=None if raw["generator"] is None else str(raw["generator"]),
        dgp_seed=None if raw["dgp_seed"] is None else int(raw["dgp_seed"]),
        noise_level=None if raw["noise_level"] is None else float(raw["noise_level"]),
        source=None if raw["source"] is None else str(raw["source"]),
        frequency=None if raw["frequency"] is None else str(raw["frequency"]),
        series_id=None if raw["series_id"] is None else str(raw["series_id"]),
        optimizer_seed=None if raw["optimizer_seed"] is None else int(raw["optimizer_seed"]),
        seasonal_period=int(raw["seasonal_period"]),
        n_lags=int(raw["n_lags"]),
        radius=None if raw["radius"] is None else float(raw["radius"]),
        alpha=None if raw["alpha"] is None else float(raw["alpha"]),
    )
    return execute_final_task(
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
    preflight: Any,
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
        "protocol_fingerprint": str(preflight.protocol_fingerprint),
        "manifest_sha256": str(preflight.manifest_sha256),
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
        },
        "result_state_at_freeze": {
            "completed": 0,
            "successful": 0,
            "failed": 0,
            "pending": int(preflight.task_count),
            "final_outcomes_generated": False,
        },
        "execution_gate": {
            "required_tag": FINAL_EXECUTION_FREEZE_TAG,
            "required_tag_target": str(preflight.execution_code_commit),
            "tag_required_before_any_task_execution": True,
        },
        "lock_rule": (
            "After this workspace freeze is created, Final execution is permitted only "
            "from this exact execution_code_commit and only after the required annotated "
            "execution-freeze tag points to that commit. No code/protocol/manifest change "
            "is admissible in response to observed Final outcomes."
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
        raise FinalParallelError("Final execution workspace freeze already exists.")
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
        raise FinalParallelError("Final execution workspace freeze is missing.")
    payload = freeze_path.read_bytes()
    expected = sha_path.read_text(encoding="ascii").strip()
    actual = _sha256_bytes(payload)
    if actual != expected:
        raise FinalParallelError("Final execution workspace freeze SHA-256 mismatch.")
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalParallelError("Final execution workspace freeze is invalid JSON.") from exc
    if data.get("schema_version") != EXECUTION_FREEZE_SCHEMA:
        raise FinalParallelError("Final execution workspace freeze schema mismatch.")
    if data.get("status") != EXECUTION_FREEZE_STATUS:
        raise FinalParallelError("Final execution workspace freeze is not locked.")
    state = data.get("result_state_at_freeze", {})
    if (
        int(state.get("completed", -1)) != 0
        or int(state.get("successful", -1)) != 0
        or int(state.get("failed", -1)) != 0
        or bool(state.get("final_outcomes_generated", True))
    ):
        raise FinalParallelError("Execution workspace was not frozen before Final outcomes.")
    return data


def _result_files(workspace: str | os.PathLike[str]) -> tuple[Path, ...]:
    results_dir = Path(workspace) / "results"
    if not results_dir.exists():
        return ()
    if not results_dir.is_dir():
        raise FinalParallelError("Final results path exists but is not a directory.")
    unexpected = [
        p for p in results_dir.iterdir()
        if not (p.is_file() and p.suffix == ".json" and len(p.stem) == 64)
    ]
    if unexpected:
        names = ", ".join(sorted(p.name for p in unexpected)[:10])
        raise FinalParallelError(
            f"Unexpected entries in Final results directory: {names}"
        )
    return tuple(sorted(results_dir.glob("*.json")))


def inspect_final_workspace_results(
    workspace: str | os.PathLike[str],
    manifest: Any,
) -> tuple[int, int, int, int]:
    from . import final_execution as fe

    task_by_id = {task.run_id: task for task in manifest.tasks}
    if len(task_by_id) != manifest.task_count:
        raise FinalParallelError("Frozen Final manifest contains duplicate run IDs.")
    completed = 0
    successful = 0
    failed = 0
    for path in _result_files(workspace):
        if path.stem not in task_by_id:
            raise FinalParallelError(
                f"Result file is not part of frozen Final manifest: {path.name}"
            )
        result = fe.load_final_result(path)
        if result.task != task_by_id[path.stem]:
            raise FinalParallelError(
                f"Stored Final result task mismatch: {path.name}"
            )
        completed += 1
        if result.successful:
            successful += 1
        else:
            failed += 1
    pending = int(manifest.task_count) - completed
    if pending < 0:
        raise FinalParallelError("Final workspace has more results than frozen tasks.")
    return completed, successful, failed, pending


def prepare_final_execution_workspace(
    workspace: str | os.PathLike[str] = r"outputs\v2\final_external",
    *,
    project_root: str | os.PathLike[str] = ".",
    workers: int = DEFAULT_WORKERS,
) -> tuple[FinalWorkspaceState, bool]:
    workers = _validate_workers(workers)

    from . import final_execution as fe

    preflight = fe.verify_final_execution_preflight(
        workspace,
        project_root=project_root,
    )
    manifest = fe.load_frozen_manifest(workspace)
    completed, successful, failed, pending = inspect_final_workspace_results(
        workspace, manifest
    )
    if completed != 0:
        raise FinalParallelError(
            "Cannot create/verify the pre-Final execution workspace with existing results."
        )

    out = Path(workspace)
    freeze_path = out / EXECUTION_FREEZE_FILENAME
    sha_path = out / EXECUTION_FREEZE_SHA_FILENAME
    created = False
    if freeze_path.exists() or sha_path.exists():
        if not (freeze_path.is_file() and sha_path.is_file()):
            raise FinalParallelError("Incomplete Final execution workspace freeze.")
        freeze = load_execution_workspace_freeze(out)
    else:
        freeze = _execution_freeze_payload(preflight, workers=workers)
        write_execution_workspace_freeze(out, freeze)
        freeze = load_execution_workspace_freeze(out)
        created = True

    expected_commit = str(freeze["execution_code_commit"])
    if expected_commit != preflight.execution_code_commit:
        raise FinalParallelError(
            "Current Git HEAD differs from the frozen Final execution code commit."
        )
    if int(freeze["task_count"]) != preflight.task_count:
        raise FinalParallelError("Execution workspace task-count freeze mismatch.")
    if str(freeze["manifest_sha256"]) != preflight.manifest_sha256:
        raise FinalParallelError("Execution workspace manifest SHA-256 freeze mismatch.")
    if str(freeze["protocol_fingerprint"]) != preflight.protocol_fingerprint:
        raise FinalParallelError("Execution workspace protocol fingerprint freeze mismatch.")
    if float(freeze["selected_radius"]) != preflight.selected_radius:
        raise FinalParallelError("Execution workspace radius freeze mismatch.")
    if float(freeze["selected_alpha"]) != preflight.selected_alpha:
        raise FinalParallelError("Execution workspace alpha freeze mismatch.")

    return (
        FinalWorkspaceState(
            execution_code_commit=preflight.execution_code_commit,
            manifest_code_commit=preflight.manifest_code_commit,
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


def verify_final_execution_workspace(
    workspace: str | os.PathLike[str] = r"outputs\v2\final_external",
    *,
    project_root: str | os.PathLike[str] = ".",
) -> FinalWorkspaceState:
    from . import final_execution as fe

    preflight = fe.verify_final_execution_preflight(
        workspace,
        project_root=project_root,
    )
    freeze = load_execution_workspace_freeze(workspace)
    if str(freeze["execution_code_commit"]) != preflight.execution_code_commit:
        raise FinalParallelError(
            "Current Git HEAD differs from the frozen Final execution code commit."
        )
    if str(freeze["manifest_code_commit"]) != preflight.manifest_code_commit:
        raise FinalParallelError("Frozen manifest-code commit mismatch.")
    if str(freeze["protocol_fingerprint"]) != preflight.protocol_fingerprint:
        raise FinalParallelError("Frozen protocol fingerprint mismatch.")
    if str(freeze["manifest_sha256"]) != preflight.manifest_sha256:
        raise FinalParallelError("Frozen manifest SHA-256 mismatch.")
    if int(freeze["task_count"]) != preflight.task_count:
        raise FinalParallelError("Frozen task count mismatch.")
    if float(freeze["selected_radius"]) != preflight.selected_radius:
        raise FinalParallelError("Frozen selected radius mismatch.")
    if float(freeze["selected_alpha"]) != preflight.selected_alpha:
        raise FinalParallelError("Frozen selected alpha mismatch.")

    manifest = fe.load_frozen_manifest(workspace)
    completed, successful, failed, pending = inspect_final_workspace_results(
        workspace, manifest
    )
    return FinalWorkspaceState(
        execution_code_commit=preflight.execution_code_commit,
        manifest_code_commit=preflight.manifest_code_commit,
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
        target = _git(root, "rev-list", "-n", "1", FINAL_EXECUTION_FREEZE_TAG)
    except FinalParallelError as exc:
        raise FinalParallelError(
            f"Required Final execution-freeze tag is missing: {FINAL_EXECUTION_FREEZE_TAG}"
        ) from exc
    if target != expected:
        raise FinalParallelError(
            f"Final execution-freeze tag target mismatch: expected {expected}, got {target}."
        )


def _schedule_selected_tasks(
    selected: Sequence[Any],
    *,
    workspace: str | os.PathLike[str],
    project_root: str | os.PathLike[str],
    workers: int,
    fail_fast: bool,
    progress_callback: Callable[[int, int, Any], None] | None,
    executor_factory: Callable[[int], Any],
    worker_function: Callable[[Mapping[str, object]], Any],
) -> tuple[int, int, int, bool, float]:
    from . import final_execution as fe

    selected = tuple(selected)
    if not selected:
        return 0, 0, 0, False, 0.0

    max_in_flight = max(workers, workers * DEFAULT_IN_FLIGHT_MULTIPLIER)
    started = time.perf_counter()
    completed_this_call = 0
    successful_this_call = 0
    failed_this_call = 0
    stopped_after_failure = False

    task_iter = iter(selected)
    in_flight: dict[Any, Any] = {}

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
            # Stable parent-side handling for futures that complete in the same wait().
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
                    raise FinalParallelError(
                        "Worker process raised outside the frozen Final task-level "
                        f"failure contract for run {task.run_id}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                if result.task != task:
                    for other in in_flight:
                        other.cancel()
                    raise FinalParallelError(
                        "Worker returned a result for the wrong frozen Final task."
                    )

                fe.validate_final_result(result)
                fe.write_final_result(workspace, result)
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

            # A task-level scientific failure is checkpointed. No new work is
            # submitted, but already-running tasks are drained and checkpointed so
            # no generated Final outcomes are silently discarded.
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


def run_pending_final_tasks_parallel(
    workspace: str | os.PathLike[str] = r"outputs\v2\final_external",
    *,
    project_root: str | os.PathLike[str] = ".",
    workers: int = DEFAULT_WORKERS,
    max_runs: int | None = None,
    progress_callback: Callable[[int, int, Any], None] | None = None,
    _executor_factory: Callable[[int], Any] | None = None,
    _worker_function: Callable[[Mapping[str, object]], Any] | None = None,
) -> FinalParallelSummary:
    workers = _validate_workers(workers)
    max_runs = _validate_max_runs(max_runs)

    from . import final_execution as fe

    freeze_exists = (
        (Path(workspace) / EXECUTION_FREEZE_FILENAME).exists()
        or (Path(workspace) / EXECUTION_FREEZE_SHA_FILENAME).exists()
    )

    if not freeze_exists:
        if max_runs != 0:
            raise FinalParallelError(
                "Final execution workspace has not been frozen. First run exactly "
                "with --max-runs 0, inspect/back up the freeze, and create the "
                f"{FINAL_EXECUTION_FREEZE_TAG} tag before any Final task."
            )
        before, created = prepare_final_execution_workspace(
            workspace,
            project_root=project_root,
            workers=workers,
        )
        if not created:
            raise FinalParallelError("Expected to create the Final execution workspace freeze.")
    else:
        before = verify_final_execution_workspace(
            workspace,
            project_root=project_root,
        )

    if before.failed_count:
        raise FinalParallelError(
            "Fail-closed gate: the Final workspace already contains failed runs "
            f"({before.failed_count}); refusing to schedule additional Final tasks."
        )

    manifest = fe.load_frozen_manifest(workspace)
    pending = fe.pending_final_tasks(workspace, manifest)

    if max_runs is not None:
        pending = pending[:max_runs]
    selected_count = len(pending)
    max_in_flight = max(workers, workers * DEFAULT_IN_FLIGHT_MULTIPLIER)

    if selected_count == 0:
        return FinalParallelSummary(
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

    # The workspace freeze can be prepared without the tag, but the first actual
    # Final outcome is impossible until the exact execution commit is tagged.
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

    after = verify_final_execution_workspace(
        workspace,
        project_root=project_root,
    )
    expected_completed = before.completed_count + completed_this_call
    if after.completed_count != expected_completed:
        raise FinalParallelError(
            "Post-run Final workspace count mismatch: expected "
            f"{expected_completed}, observed {after.completed_count}."
        )
    if after.successful_count != before.successful_count + successful_this_call:
        raise FinalParallelError("Post-run successful-count mismatch.")
    if after.failed_count != before.failed_count + failed_this_call:
        raise FinalParallelError("Post-run failed-count mismatch.")

    return FinalParallelSummary(
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
    "FINAL_EXECUTION_FREEZE_TAG",
    "FinalParallelError",
    "FinalWorkspaceState",
    "FinalParallelSummary",
    "prepare_final_execution_workspace",
    "verify_final_execution_workspace",
    "assert_execution_freeze_tag",
    "inspect_final_workspace_results",
    "run_pending_final_tasks_parallel",
]
