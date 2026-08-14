from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

import nfpsosc.v2.ablation_parallel as ap
import nfpsosc.v2.ablation_execution as ae


def _preflight():
    return ae.AblationExecutionPreflight(
        execution_code_commit="a" * 40,
        manifest_code_commit=ae.FROZEN_MANIFEST_CODE_COMMIT,
        manifest_freeze_commit=ae.FROZEN_MANIFEST_FREEZE_COMMIT,
        protocol_fingerprint=ae.FROZEN_PROTOCOL_FINGERPRINT,
        manifest_sha256=ae.FROZEN_MANIFEST_SHA256,
        external_final_results_aggregate_sha256=ae.FROZEN_EXTERNAL_FINAL_AGGREGATE,
        task_count=5250,
        selected_radius=1.0,
        selected_alpha=0.05,
    )


class _ImmediateExecutor:
    def __init__(self, workers):
        self.workers = workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, payload):
        future = Future()
        try:
            future.set_result(fn(payload))
        except Exception as exc:
            future.set_exception(exc)
        return future


def test_default_worker_policy_is_single_process_after_memory_incident():
    assert ap.DEFAULT_WORKERS == 1
    assert ap.DEFAULT_IN_FLIGHT_MULTIPLIER == 2
    assert ap.THREADS_PER_WORKER == 1


def test_execution_freeze_payload_is_pre_outcome_and_exact():
    payload = ap._execution_freeze_payload(_preflight(), workers=1)
    assert payload["status"] == "LOCKED_PRE_ABLATION_OUTCOMES"
    assert payload["task_count"] == 5250
    assert payload["parallel_policy"]["default_workers"] == 1
    assert payload["parallel_policy"]["max_in_flight"] == 2
    assert payload["parallel_policy"]["frozen_worker_count_enforced"] is True
    assert payload["result_state_at_freeze"] == {
        "completed": 0,
        "successful": 0,
        "failed": 0,
        "pending": 5250,
        "ablation_outcomes_generated": False,
    }


def test_execution_freeze_roundtrip_and_hash_guard(tmp_path):
    payload = ap._execution_freeze_payload(_preflight(), workers=1)
    freeze, sidecar = ap.write_execution_workspace_freeze(tmp_path, payload)
    assert ap.load_execution_workspace_freeze(tmp_path) == payload

    freeze.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ap.AblationParallelError, match="SHA-256 mismatch"):
        ap.load_execution_workspace_freeze(tmp_path)


def test_invalid_workers_rejected():
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ap.AblationParallelError):
            ap._validate_workers(bad)


def test_invalid_max_runs_rejected():
    for bad in (-1, True, 1.5):
        with pytest.raises(ap.AblationParallelError):
            ap._validate_max_runs(bad)


def test_task_payload_preserves_ablation_identity():
    manifest = ae.load_frozen_ablation_manifest(project_root=".")
    task = manifest.tasks[0]
    payload = ap._task_payload(task, ".")
    raw = payload["task"]
    assert raw["run_id"] == task.run_id
    assert raw["variant"] == task.variant
    assert raw["optimizer_seed"] == task.optimizer_seed
    assert raw["radius"] == 1.0
    assert raw["alpha"] == 0.05


def test_fresh_workspace_inspection_is_zero_completed(tmp_path):
    manifest = ae.load_frozen_ablation_manifest(project_root=".")
    assert ap.inspect_ablation_workspace_results(
        tmp_path,
        manifest,
        project_root=".",
    ) == (0, 0, 0, 5250)


def test_unknown_result_file_is_rejected(tmp_path):
    manifest = ae.load_frozen_ablation_manifest(project_root=".")
    results = tmp_path / "results"
    results.mkdir()
    (results / ("f" * 64 + ".json")).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ap.AblationParallelError, match="not part"):
        ap.inspect_ablation_workspace_results(
            tmp_path,
            manifest,
            project_root=".",
        )


def test_scheduler_parent_checkpoints_fake_successes(monkeypatch, tmp_path):
    manifest = ae.load_frozen_ablation_manifest(project_root=".")
    selected = manifest.tasks[:3]
    by_id = {task.run_id: task for task in selected}
    written = []

    monkeypatch.setattr(ae, "validate_ablation_result", lambda result, project_root=".": None)
    monkeypatch.setattr(
        ae,
        "write_ablation_result",
        lambda workspace, result, project_root=".": written.append(result.task.run_id),
    )

    def worker(payload):
        run_id = payload["task"]["run_id"]
        return SimpleNamespace(task=by_id[run_id], successful=True)

    observed = ap._schedule_selected_tasks(
        selected,
        workspace=tmp_path,
        project_root=".",
        workers=1,
        fail_fast=True,
        progress_callback=None,
        executor_factory=lambda workers: _ImmediateExecutor(workers),
        worker_function=worker,
    )

    assert observed[:4] == (3, 3, 0, False)
    assert sorted(written) == sorted(by_id)


def test_scheduler_stops_new_submission_after_failure_but_drains_inflight(
    monkeypatch,
    tmp_path,
):
    manifest = ae.load_frozen_ablation_manifest(project_root=".")
    selected = manifest.tasks[:4]
    by_id = {task.run_id: task for task in selected}
    written = []

    monkeypatch.setattr(ae, "validate_ablation_result", lambda result, project_root=".": None)
    monkeypatch.setattr(
        ae,
        "write_ablation_result",
        lambda workspace, result, project_root=".": written.append(result.task.run_id),
    )

    first_two = {task.run_id for task in selected[:2]}
    failing = selected[0].run_id

    def worker(payload):
        run_id = payload["task"]["run_id"]
        return SimpleNamespace(
            task=by_id[run_id],
            successful=(run_id != failing),
        )

    observed = ap._schedule_selected_tasks(
        selected,
        workspace=tmp_path,
        project_root=".",
        workers=1,
        fail_fast=True,
        progress_callback=None,
        executor_factory=lambda workers: _ImmediateExecutor(workers),
        worker_function=worker,
    )

    # max_in_flight=2: both initially submitted tasks are drained, no third/fourth.
    assert observed[0] == 2
    assert observed[1] == 1
    assert observed[2] == 1
    assert observed[3] is True
    assert set(written) == first_two
