"""
Bounded multi-process execution layer for the frozen V2 development runner.

Scientific task definition, model fitting, metrics, checkpoints, and selection remain
owned by ``development_runner.py``.  This module changes execution scheduling only.

Design goals
------------
* four workers by default (chosen from a performance-blind debug benchmark);
* one numerical-library thread per worker to avoid BLAS/OpenMP oversubscription;
* spawn-based worker processes on Windows and other platforms;
* primitive task payloads so worker thread caps are installed before importing
  NumPy/scikit-learn through ``development_runner``;
* only the parent process writes result checkpoints;
* bounded in-flight work;
* fail-closed behavior when a stored or newly completed run has failed;
* resume through the existing per-run checkpoint contract;
* mandatory workspace/code/protocol/environment verification before scheduling.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping


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


class DevelopmentParallelError(RuntimeError):
    """Raised when parallel development execution violates the execution contract."""


@dataclass(frozen=True)
class ParallelRunSummary:
    workers: int
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
        raise DevelopmentParallelError("workers must be a positive integer.")
    return int(workers)


def _validate_max_runs(max_runs: int | None) -> int | None:
    if max_runs is None:
        return None
    if isinstance(max_runs, bool) or not isinstance(max_runs, int) or max_runs < 0:
        raise DevelopmentParallelError(
            "max_runs must be a non-negative integer or None."
        )
    return int(max_runs)


def _configure_worker_threads() -> None:
    value = str(THREADS_PER_WORKER)
    for name in THREAD_ENV_NAMES:
        os.environ[name] = value


def _task_payload(task: Any) -> dict[str, object]:
    return {
        "run_id": str(task.run_id),
        "generator": str(task.generator),
        "dgp_seed": int(task.dgp_seed),
        "noise_level": float(task.noise_level),
        "optimizer_seed": int(task.optimizer_seed),
        "radius": float(task.radius),
        "alpha": float(task.alpha),
    }


def _execute_task_payload(payload: Mapping[str, object]):
    # The worker initializer runs before tasks are dequeued. Importing the
    # scientific runner here ensures NumPy/BLAS imports see the thread caps.
    from .development_runner import DevelopmentRunTask, execute_development_task

    task = DevelopmentRunTask(
        run_id=str(payload["run_id"]),
        generator=str(payload["generator"]),
        dgp_seed=int(payload["dgp_seed"]),
        noise_level=float(payload["noise_level"]),
        optimizer_seed=int(payload["optimizer_seed"]),
        radius=float(payload["radius"]),
        alpha=float(payload["alpha"]),
    )
    return execute_development_task(task)


def _production_executor(max_workers: int):
    context = mp.get_context("spawn")
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
        initializer=_configure_worker_threads,
    )


def run_pending_development_tasks_parallel(
    output_dir: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str],
    workers: int = DEFAULT_WORKERS,
    max_runs: int | None = None,
    fail_fast: bool = True,
    progress_callback: Callable[[int, int, Any], None] | None = None,
    _executor_factory: Callable[[int], Any] | None = None,
    _worker_function: Callable[[Mapping[str, object]], Any] | None = None,
) -> ParallelRunSummary:
    """
    Execute pending frozen development tasks with bounded multi-processing.

    The private ``_executor_factory`` and ``_worker_function`` hooks exist only
    for deterministic unit testing; production callers must leave them unset.
    """
    workers = _validate_workers(workers)
    max_runs = _validate_max_runs(max_runs)
    if not isinstance(fail_fast, bool):
        raise DevelopmentParallelError("fail_fast must be boolean.")

    from . import development_runner as dr
    from .development_workspace import verify_development_workspace

    verification_before = verify_development_workspace(
        output_dir,
        project_root=project_root,
        require_no_results=False,
    )
    if verification_before.failed_count:
        raise DevelopmentParallelError(
            "A007 fail-closed gate: the workspace already contains failed "
            f"development runs ({verification_before.failed_count}); refusing "
            "to schedule additional runs."
        )

    manifest = dr.load_development_manifest(
        output_dir,
        expected_protocol_fingerprint=verification_before.protocol_fingerprint,
        expected_code_commit=verification_before.code_commit,
    )
    pending = dr.pending_development_tasks(output_dir, manifest)
    if max_runs is not None:
        pending = pending[:max_runs]

    selected_count = len(pending)
    if selected_count == 0:
        return ParallelRunSummary(
            workers=workers,
            selected_count=0,
            completed_this_call=0,
            successful_this_call=0,
            failed_this_call=0,
            completed_total=verification_before.completed_count,
            successful_total=verification_before.successful_count,
            failed_total=verification_before.failed_count,
            pending_total=verification_before.pending_count,
            wall_seconds=0.0,
            stopped_after_failure=False,
        )

    executor_factory = _executor_factory or _production_executor
    worker_function = _worker_function or _execute_task_payload
    max_in_flight = max(workers, workers * DEFAULT_IN_FLIGHT_MULTIPLIER)

    started = time.perf_counter()
    completed_this_call = 0
    successful_this_call = 0
    failed_this_call = 0
    stopped_after_failure = False

    task_iter = iter(pending)
    in_flight: dict[Any, Any] = {}

    def submit_next(executor: Any) -> bool:
        try:
            task = next(task_iter)
        except StopIteration:
            return False
        future = executor.submit(worker_function, _task_payload(task))
        in_flight[future] = task
        return True

    with executor_factory(workers) as executor:
        while len(in_flight) < max_in_flight and submit_next(executor):
            pass

        while in_flight:
            done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
            for future in done:
                task = in_flight.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    for other in in_flight:
                        other.cancel()
                    raise DevelopmentParallelError(
                        "Worker process raised outside the frozen task-level failure "
                        f"contract for run {task.run_id}: {type(exc).__name__}: {exc}"
                    ) from exc

                if result.task != task:
                    for other in in_flight:
                        other.cancel()
                    raise DevelopmentParallelError(
                        "Worker returned a result for the wrong development task."
                    )

                dr.validate_development_result(result)
                dr.write_development_result(output_dir, result)

                completed_this_call += 1
                if result.successful:
                    successful_this_call += 1
                else:
                    failed_this_call += 1
                    stopped_after_failure = True

                if progress_callback is not None:
                    progress_callback(
                        completed_this_call,
                        selected_count,
                        result,
                    )

            # After a scientific/task-level failure, checkpoint already-running
            # jobs but stop adding new work. A007 blocks selection anyway.
            if not (fail_fast and stopped_after_failure):
                while len(in_flight) < max_in_flight and submit_next(executor):
                    pass

    wall_seconds = time.perf_counter() - started

    verification_after = verify_development_workspace(
        output_dir,
        project_root=project_root,
        require_no_results=False,
    )
    expected_completed_total = (
        verification_before.completed_count + completed_this_call
    )
    if verification_after.completed_count != expected_completed_total:
        raise DevelopmentParallelError(
            "Post-run workspace count mismatch: expected "
            f"{expected_completed_total} completed results, observed "
            f"{verification_after.completed_count}."
        )
    if verification_after.failed_count != (
        verification_before.failed_count + failed_this_call
    ):
        raise DevelopmentParallelError(
            "Post-run failed-count mismatch after parallel execution."
        )

    return ParallelRunSummary(
        workers=workers,
        selected_count=selected_count,
        completed_this_call=completed_this_call,
        successful_this_call=successful_this_call,
        failed_this_call=failed_this_call,
        completed_total=verification_after.completed_count,
        successful_total=verification_after.successful_count,
        failed_total=verification_after.failed_count,
        pending_total=verification_after.pending_count,
        wall_seconds=float(wall_seconds),
        stopped_after_failure=stopped_after_failure,
    )


__all__ = [
    "DEFAULT_WORKERS",
    "DEFAULT_IN_FLIGHT_MULTIPLIER",
    "THREADS_PER_WORKER",
    "THREAD_ENV_NAMES",
    "DevelopmentParallelError",
    "ParallelRunSummary",
    "run_pending_development_tasks_parallel",
]
