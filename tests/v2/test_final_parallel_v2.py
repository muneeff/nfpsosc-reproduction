from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

import pytest

import nfpsosc.v2.final_execution as fe
import nfpsosc.v2.final_parallel as fp


class _Preflight:
    execution_code_commit = "1" * 40
    manifest_code_commit = fe.FROZEN_MANIFEST_CODE_COMMIT
    protocol_fingerprint = fe.FROZEN_PROTOCOL_FINGERPRINT
    manifest_sha256 = fe.FROZEN_MANIFEST_SHA256
    task_count = 8780
    selected_radius = 1.0
    selected_alpha = 0.05


def _copy_manifest(tmp_path: Path) -> Path:
    source = Path("outputs") / "v2" / "final_external"
    (tmp_path / "final_external_manifest.json").write_bytes(
        (source / "final_external_manifest.json").read_bytes()
    )
    (tmp_path / "final_external_manifest.sha256").write_bytes(
        (source / "final_external_manifest.sha256").read_bytes()
    )
    return tmp_path


def _success_result(task):
    metrics = fe.FinalMetrics(
        mase=1.0,
        mae=1.0,
        rmse=1.0,
        smape=10.0,
        rmsse=1.0,
        mase_scale=1.0,
        rmsse_scale_sq=1.0,
    )
    if task.method == "pc_nfpso":
        return fe.FinalRunResult(
            task=task,
            status="success",
            metrics=metrics,
            y_true=(1.0,),
            y_pred=(1.0,),
            validation_size=8,
            best_objective=0.5,
            n_rules=2,
            best_position_sha256="a" * 64,
            selected_config={"toy": True},
            selection_failures=(),
            failure=None,
        )
    return fe.FinalRunResult(
        task=task,
        status="success",
        metrics=metrics,
        y_true=(1.0,),
        y_pred=(1.0,),
        validation_size=None,
        best_objective=None,
        n_rules=None,
        best_position_sha256=None,
        selected_config={"toy": True},
        selection_failures=(),
        failure=None,
    )


def _failed_result(task):
    return fe.FinalRunResult(
        task=task,
        status="failed",
        metrics=None,
        y_true=(),
        y_pred=(),
        validation_size=None,
        best_objective=None,
        n_rules=None,
        best_position_sha256=None,
        selected_config={},
        selection_failures=(),
        failure=fe.FinalRunFailure(
            stage="load",
            error_type="ToyFailure",
            message="toy",
            component_stage="test",
            origin=None,
        ),
    )


def test_default_parallel_policy_matches_frozen_plan() -> None:
    assert fp.DEFAULT_WORKERS == 4
    assert fp.DEFAULT_IN_FLIGHT_MULTIPLIER == 2
    assert fp.THREADS_PER_WORKER == 1
    assert fp.FINAL_EXECUTION_FREEZE_TAG == "v2-final-execution-freeze-2026-08-14"


def test_worker_thread_caps_cover_major_numeric_runtimes(monkeypatch) -> None:
    for name in fp.THREAD_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    fp._configure_worker_threads()
    assert all(os.environ[name] == "1" for name in fp.THREAD_ENV_NAMES)


def test_execution_freeze_payload_is_pre_outcome_and_exact() -> None:
    payload = fp._execution_freeze_payload(_Preflight(), workers=4)
    assert payload["schema_version"] == fp.EXECUTION_FREEZE_SCHEMA
    assert payload["status"] == fp.EXECUTION_FREEZE_STATUS
    assert payload["execution_code_commit"] == "1" * 40
    assert payload["manifest_code_commit"] == fe.FROZEN_MANIFEST_CODE_COMMIT
    assert payload["protocol_fingerprint"] == fe.FROZEN_PROTOCOL_FINGERPRINT
    assert payload["manifest_sha256"] == fe.FROZEN_MANIFEST_SHA256
    assert payload["task_count"] == 8780
    assert payload["selected_radius"] == 1.0
    assert payload["selected_alpha"] == 0.05
    assert payload["parallel_policy"]["default_workers"] == 4
    assert payload["parallel_policy"]["max_in_flight"] == 8
    assert payload["result_state_at_freeze"]["completed"] == 0
    assert payload["result_state_at_freeze"]["final_outcomes_generated"] is False


def test_execution_freeze_roundtrip_and_hash_guard(tmp_path: Path) -> None:
    payload = fp._execution_freeze_payload(_Preflight(), workers=4)
    freeze_path, sha_path = fp.write_execution_workspace_freeze(tmp_path, payload)
    expected = sha_path.read_text(encoding="ascii").strip()
    assert hashlib.sha256(freeze_path.read_bytes()).hexdigest() == expected
    loaded = fp.load_execution_workspace_freeze(tmp_path)
    assert loaded == payload
    with pytest.raises(fp.FinalParallelError, match="already exists"):
        fp.write_execution_workspace_freeze(tmp_path, payload)
    freeze_path.write_bytes(freeze_path.read_bytes() + b" ")
    with pytest.raises(fp.FinalParallelError, match="SHA-256"):
        fp.load_execution_workspace_freeze(tmp_path)


def test_fresh_manifest_copy_has_zero_results_and_8780_pending(tmp_path: Path) -> None:
    workspace = _copy_manifest(tmp_path)
    manifest = fe.load_frozen_manifest(workspace)
    completed, successful, failed, pending = fp.inspect_final_workspace_results(
        workspace, manifest
    )
    assert (completed, successful, failed, pending) == (0, 0, 0, 8780)


def test_unknown_result_file_is_rejected(tmp_path: Path) -> None:
    workspace = _copy_manifest(tmp_path)
    results = workspace / "results"
    results.mkdir()
    (results / ("f" * 64 + ".json")).write_text("{}\n", encoding="utf-8")
    manifest = fe.load_frozen_manifest(workspace)
    with pytest.raises(fp.FinalParallelError, match="not part"):
        fp.inspect_final_workspace_results(workspace, manifest)


def test_unexpected_temp_or_nonjson_result_entry_is_rejected(tmp_path: Path) -> None:
    workspace = _copy_manifest(tmp_path)
    results = workspace / "results"
    results.mkdir()
    (results / ".orphan.tmp").write_text("x", encoding="utf-8")
    manifest = fe.load_frozen_manifest(workspace)
    with pytest.raises(fp.FinalParallelError, match="Unexpected"):
        fp.inspect_final_workspace_results(workspace, manifest)


def test_scheduler_parent_writes_success_checkpoints_with_fake_workers(tmp_path: Path) -> None:
    workspace = _copy_manifest(tmp_path)
    manifest = fe.load_frozen_manifest(workspace)
    selected = manifest.tasks[:4]
    by_run = {task.run_id: task for task in selected}

    def worker(payload):
        return _success_result(by_run[payload["task"]["run_id"]])

    completed, successful, failed, stopped, wall = fp._schedule_selected_tasks(
        selected,
        workspace=workspace,
        project_root=".",
        workers=2,
        fail_fast=True,
        progress_callback=None,
        executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
        worker_function=worker,
    )
    assert completed == 4
    assert successful == 4
    assert failed == 0
    assert stopped is False
    assert wall >= 0.0
    assert len(list((workspace / "results").glob("*.json"))) == 4
    c, s, f, p = fp.inspect_final_workspace_results(workspace, manifest)
    assert (c, s, f, p) == (4, 4, 0, 8776)


def test_scheduler_stops_new_submissions_after_failure_but_drains_inflight(tmp_path: Path) -> None:
    workspace = _copy_manifest(tmp_path)
    manifest = fe.load_frozen_manifest(workspace)
    selected = manifest.tasks[:12]
    first_run = selected[0].run_id
    by_run = {task.run_id: task for task in selected}

    def worker(payload):
        task = by_run[payload["task"]["run_id"]]
        if task.run_id == first_run:
            return _failed_result(task)
        time.sleep(0.02)
        return _success_result(task)

    completed, successful, failed, stopped, _ = fp._schedule_selected_tasks(
        selected,
        workspace=workspace,
        project_root=".",
        workers=2,
        fail_fast=True,
        progress_callback=None,
        executor_factory=lambda n: ThreadPoolExecutor(max_workers=n),
        worker_function=worker,
    )
    # Up to max_in_flight=4 were submitted initially. No more are submitted
    # once the first failure is observed; all already submitted work is recorded.
    assert 1 <= completed <= 4
    assert failed == 1
    assert successful == completed - 1
    assert stopped is True
    assert len(list((workspace / "results").glob("*.json"))) == completed


def test_task_payload_roundtrips_all_frozen_identity_fields() -> None:
    task = fe.load_frozen_manifest(r"outputs\v2\final_external").tasks[0]
    payload = fp._task_payload(task, ".")
    raw = payload["task"]
    assert raw["run_id"] == task.run_id
    assert raw["domain"] == task.domain
    assert raw["method"] == task.method
    assert raw["dgp_seed"] == task.dgp_seed
    assert raw["optimizer_seed"] == task.optimizer_seed
    assert raw["radius"] == task.radius
    assert raw["alpha"] == task.alpha


@pytest.mark.parametrize("value", [0, -1, True])
def test_invalid_workers_rejected(value) -> None:
    with pytest.raises(fp.FinalParallelError):
        fp._validate_workers(value)


@pytest.mark.parametrize("value", [-1, True])
def test_invalid_max_runs_rejected(value) -> None:
    with pytest.raises(fp.FinalParallelError):
        fp._validate_max_runs(value)


def test_result_checkpoint_overwrite_remains_forbidden_under_scheduler(tmp_path: Path) -> None:
    workspace = _copy_manifest(tmp_path)
    task = fe.load_frozen_manifest(workspace).tasks[0]
    result = _success_result(task)
    fe.write_final_result(workspace, result)
    with pytest.raises(fe.FinalExecutionError, match="already exists"):
        fe.write_final_result(workspace, result)
