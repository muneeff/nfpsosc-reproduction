from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import nfpsosc.v2.development_parallel as dp
import nfpsosc.v2.development_runner as dr
import nfpsosc.v2.development_workspace as dw


def _manifest_with_tasks(tasks):
    base = dr.build_development_manifest(
        protocol_fingerprint_value="a" * 64,
        code_commit="b" * 40,
    )
    return dr.DevelopmentManifest(
        schema_version=base.schema_version,
        protocol_fingerprint=base.protocol_fingerprint,
        code_commit=base.code_commit,
        task_count=base.task_count,
        condition_count=base.condition_count,
        generators=base.generators,
        dgp_seeds=base.dgp_seeds,
        noise_levels=base.noise_levels,
        optimizer_seeds=base.optimizer_seeds,
        candidate_pairs=base.candidate_pairs,
        tasks=base.tasks,
    )


def _success(task: dr.DevelopmentRunTask):
    m = dr.seasonal_period(task.generator)
    L = dr.frozen_lag_dimension(m)
    V = dr.pc_nfpso_validation_size(dr.PRETEST_LENGTH, L)
    return dr.DevelopmentRunResult(
        task=task,
        status="success",
        mase=1.0,
        mase_scale=1.0,
        n_lags=L,
        validation_size=V,
        seasonal_period=m,
        best_objective=0.25,
        n_rules=3,
        failure=None,
    )


def _failed(task: dr.DevelopmentRunTask):
    m = dr.seasonal_period(task.generator)
    L = dr.frozen_lag_dimension(m)
    V = dr.pc_nfpso_validation_size(dr.PRETEST_LENGTH, L)
    return dr.DevelopmentRunResult(
        task=task,
        status="failed",
        mase=None,
        mase_scale=None,
        n_lags=L,
        validation_size=V,
        seasonal_period=m,
        best_objective=None,
        n_rules=None,
        failure=dr.DevelopmentRunFailure("fit", "RuntimeError", "forced"),
    )


def test_default_workers_and_thread_policy_are_frozen():
    assert dp.DEFAULT_WORKERS == 4
    assert dp.THREADS_PER_WORKER == 1
    assert dp.DEFAULT_IN_FLIGHT_MULTIPLIER == 2
    assert set(dp.THREAD_ENV_NAMES) == {
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    }


def test_worker_initializer_forces_single_thread(monkeypatch):
    for name in dp.THREAD_ENV_NAMES:
        monkeypatch.setenv(name, "99")
    dp._configure_worker_threads()
    assert all(
        __import__("os").environ[name] == "1"
        for name in dp.THREAD_ENV_NAMES
    )


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_worker_counts_are_rejected(workers):
    with pytest.raises(dp.DevelopmentParallelError):
        dp._validate_workers(workers)


@pytest.mark.parametrize("max_runs", [-1, True, 1.5])
def test_invalid_max_runs_are_rejected(max_runs):
    with pytest.raises(dp.DevelopmentParallelError):
        dp._validate_max_runs(max_runs)


def test_task_payload_contains_only_primitive_frozen_key_fields():
    task = dr.build_development_tasks()[0]
    payload = dp._task_payload(task)
    assert payload == {
        "run_id": task.run_id,
        "generator": task.generator,
        "dgp_seed": task.dgp_seed,
        "noise_level": task.noise_level,
        "optimizer_seed": task.optimizer_seed,
        "radius": task.radius,
        "alpha": task.alpha,
    }
    assert all(
        isinstance(v, (str, int, float))
        for v in payload.values()
    )


def test_parallel_scheduler_checkpoints_and_preserves_task_identity(
    monkeypatch, tmp_path: Path
):
    tasks = dr.build_development_tasks()[:5]
    manifest = _manifest_with_tasks(tasks)
    stored = {}

    before = SimpleNamespace(
        code_commit="b" * 40,
        protocol_fingerprint="a" * 64,
        completed_count=0,
        successful_count=0,
        failed_count=0,
        pending_count=dr.EXPECTED_RUNS,
    )

    def verify(*a, **k):
        return SimpleNamespace(
            code_commit="b" * 40,
            protocol_fingerprint="a" * 64,
            completed_count=len(stored),
            successful_count=sum(r.successful for r in stored.values()),
            failed_count=sum(not r.successful for r in stored.values()),
            pending_count=dr.EXPECTED_RUNS - len(stored),
        )

    monkeypatch.setattr(dw, "verify_development_workspace", verify)
    monkeypatch.setattr(dr, "load_development_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(
        dr,
        "pending_development_tasks",
        lambda out, m: tuple(t for t in tasks if t.run_id not in stored),
    )
    monkeypatch.setattr(dr, "validate_development_result", lambda result: None)

    def write(out, result):
        assert result.task.run_id not in stored
        stored[result.task.run_id] = result

    monkeypatch.setattr(dr, "write_development_result", write)

    def worker(payload):
        return _success(dr.DevelopmentRunTask(**payload))

    progress = []
    summary = dp.run_pending_development_tasks_parallel(
        tmp_path,
        project_root=tmp_path,
        workers=2,
        max_runs=5,
        progress_callback=lambda done, total, result: progress.append(
            (done, total, result.task.run_id)
        ),
        _executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
        _worker_function=worker,
    )
    assert summary.selected_count == 5
    assert summary.completed_this_call == 5
    assert summary.successful_this_call == 5
    assert summary.failed_this_call == 0
    assert summary.completed_total == 5
    assert summary.pending_total == dr.EXPECTED_RUNS - 5
    assert len(stored) == 5
    assert len(progress) == 5
    assert {row[2] for row in progress} == {task.run_id for task in tasks}


def test_existing_failed_workspace_blocks_new_scheduling(monkeypatch, tmp_path):
    monkeypatch.setattr(
        dw,
        "verify_development_workspace",
        lambda *a, **k: SimpleNamespace(
            code_commit="b" * 40,
            protocol_fingerprint="a" * 64,
            completed_count=1,
            successful_count=0,
            failed_count=1,
            pending_count=dr.EXPECTED_RUNS - 1,
        ),
    )
    with pytest.raises(dp.DevelopmentParallelError, match="fail-closed"):
        dp.run_pending_development_tasks_parallel(
            tmp_path,
            project_root=tmp_path,
            workers=2,
            _executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
        )


def test_fail_fast_checkpoints_failure_and_stops_refill(monkeypatch, tmp_path):
    # 20 tasks > 2*workers in-flight capacity, so fail_fast should prevent all
    # 20 from being scheduled once the first failure is observed.
    tasks = dr.build_development_tasks()[:20]
    manifest = _manifest_with_tasks(tasks)
    stored = {}

    def verify(*a, **k):
        return SimpleNamespace(
            code_commit="b" * 40,
            protocol_fingerprint="a" * 64,
            completed_count=len(stored),
            successful_count=sum(r.successful for r in stored.values()),
            failed_count=sum(not r.successful for r in stored.values()),
            pending_count=dr.EXPECTED_RUNS - len(stored),
        )

    monkeypatch.setattr(dw, "verify_development_workspace", verify)
    monkeypatch.setattr(dr, "load_development_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(
        dr,
        "pending_development_tasks",
        lambda out, m: tuple(t for t in tasks if t.run_id not in stored),
    )
    monkeypatch.setattr(dr, "validate_development_result", lambda result: None)
    monkeypatch.setattr(
        dr,
        "write_development_result",
        lambda out, result: stored.__setitem__(result.task.run_id, result),
    )

    first_id = tasks[0].run_id

    def worker(payload):
        task = dr.DevelopmentRunTask(**payload)
        return _failed(task) if task.run_id == first_id else _success(task)

    summary = dp.run_pending_development_tasks_parallel(
        tmp_path,
        project_root=tmp_path,
        workers=2,
        fail_fast=True,
        _executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
        _worker_function=worker,
    )
    assert summary.failed_this_call == 1
    assert summary.stopped_after_failure is True
    assert summary.completed_this_call <= 4
    assert len(stored) == summary.completed_this_call


def test_zero_max_runs_is_noop(monkeypatch, tmp_path):
    tasks = dr.build_development_tasks()[:3]
    manifest = _manifest_with_tasks(tasks)
    verification = SimpleNamespace(
        code_commit="b" * 40,
        protocol_fingerprint="a" * 64,
        completed_count=0,
        successful_count=0,
        failed_count=0,
        pending_count=dr.EXPECTED_RUNS,
    )
    monkeypatch.setattr(
        dw, "verify_development_workspace", lambda *a, **k: verification
    )
    monkeypatch.setattr(dr, "load_development_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(dr, "pending_development_tasks", lambda out, m: tasks)

    summary = dp.run_pending_development_tasks_parallel(
        tmp_path,
        project_root=tmp_path,
        workers=2,
        max_runs=0,
        _executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
    )
    assert summary.selected_count == 0
    assert summary.completed_this_call == 0
    assert summary.pending_total == dr.EXPECTED_RUNS
