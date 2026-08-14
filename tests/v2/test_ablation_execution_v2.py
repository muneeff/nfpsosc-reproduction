from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import nfpsosc.v2.ablation_execution as ae
import nfpsosc.v2.ablation_runner as ar
import nfpsosc.v2.final_execution as fe


def _manifest():
    return ae.load_frozen_ablation_manifest(project_root=".")


def _task(variant: str = "NF_BASE"):
    for task in _manifest().tasks:
        if task.variant == variant:
            return task
    raise AssertionError(variant)


def _metrics():
    return fe.FinalMetrics(
        mase=1.0,
        mae=2.0,
        rmse=3.0,
        smape=4.0,
        rmsse=5.0,
        mase_scale=2.0,
        rmsse_scale_sq=3.0,
    )


def test_frozen_execution_constants_match_manifest_freeze():
    assert ae.FROZEN_MANIFEST_CODE_COMMIT == "371a533f201797628106cff7845f1bf3b36f1df0"
    assert ae.FROZEN_MANIFEST_FREEZE_COMMIT == "078553e03268340af1bef5e2add47025be5de42d"
    assert ae.FROZEN_TOTAL_TASKS == 5250
    assert ae.FROZEN_MANIFEST_SHA256 == "6dedd4dd75c0953d617679226800daec5e072fb45f69af0ed3667f94a6ff376b"


def test_frozen_manifest_loads_exact_5250_new_tasks():
    manifest = _manifest()
    assert manifest.new_task_count == 5250
    assert len(manifest.tasks) == 5250
    assert all(t.variant != "PC_NFPSO" for t in manifest.tasks)


def test_pending_is_full_in_fresh_workspace(tmp_path):
    manifest = _manifest()
    pending = ae.pending_ablation_tasks(tmp_path, manifest)
    assert len(pending) == 5250
    assert pending == manifest.tasks


def test_result_path_is_immutable_per_run_checkpoint(tmp_path):
    task = _task()
    assert ae.result_path(tmp_path, task.run_id) == tmp_path / "results" / f"{task.run_id}.json"


def test_nf_base_success_result_has_no_pso_provenance(monkeypatch):
    task = _task("NF_BASE")
    monkeypatch.setattr(
        ae,
        "_frozen_task_map",
        lambda root: {task.run_id: task},
    )
    result = ae.AblationRunResult(
        task=task,
        status="success",
        metrics=_metrics(),
        y_true=(1.0, 2.0),
        y_pred=(1.1, 1.9),
        validation_size=6,
        best_objective=None,
        n_rules=2,
        best_position_sha256=None,
        selected_config={"variant": "NF_BASE", "pso": False},
        failure=None,
    )
    ae.validate_ablation_result(result)


def test_nf_base_rejects_pso_provenance(monkeypatch):
    task = _task("NF_BASE")
    monkeypatch.setattr(
        ae,
        "_frozen_task_map",
        lambda root: {task.run_id: task},
    )
    result = ae.AblationRunResult(
        task=task,
        status="success",
        metrics=_metrics(),
        y_true=(1.0,),
        y_pred=(1.0,),
        validation_size=6,
        best_objective=1.0,
        n_rules=1,
        best_position_sha256="a" * 64,
        selected_config={},
        failure=None,
    )
    with pytest.raises(ae.AblationExecutionError, match="PSO provenance"):
        ae.validate_ablation_result(result)


def test_stochastic_success_requires_optimizer_provenance(monkeypatch):
    task = _task("NFPSO")
    monkeypatch.setattr(
        ae,
        "_frozen_task_map",
        lambda root: {task.run_id: task},
    )
    result = ae.AblationRunResult(
        task=task,
        status="success",
        metrics=_metrics(),
        y_true=(1.0,),
        y_pred=(1.0,),
        validation_size=6,
        best_objective=None,
        n_rules=1,
        best_position_sha256=None,
        selected_config={},
        failure=None,
    )
    with pytest.raises(ae.AblationExecutionError, match="best objective"):
        ae.validate_ablation_result(result)


def test_result_roundtrip_and_overwrite_guard(tmp_path, monkeypatch):
    task = _task("NFPSO")
    monkeypatch.setattr(
        ae,
        "_frozen_task_map",
        lambda root: {task.run_id: task},
    )
    result = ae.AblationRunResult(
        task=task,
        status="success",
        metrics=_metrics(),
        y_true=(1.0, 2.0),
        y_pred=(1.2, 1.8),
        validation_size=6,
        best_objective=0.25,
        n_rules=3,
        best_position_sha256="b" * 64,
        selected_config={"variant": "NFPSO", "pso": True},
        failure=None,
    )
    path = ae.write_ablation_result(tmp_path, result)
    observed = ae.load_ablation_result(path)
    assert observed == result
    with pytest.raises(ae.AblationExecutionError, match="already exists"):
        ae.write_ablation_result(tmp_path, result)


def test_execute_task_rejects_nonmember_before_data_access(monkeypatch):
    task = replace(_task("NF_BASE"), run_id="f" * 64)
    monkeypatch.setattr(ae, "_frozen_task_map", lambda root: {})
    touched = {"loader": False}

    def loader(*args, **kwargs):
        touched["loader"] = True
        raise AssertionError("must not run")

    with pytest.raises(ae.AblationExecutionError, match="exact member"):
        ae.execute_ablation_task(task, series_loader=loader)
    assert touched["loader"] is False


def test_failed_result_has_no_metrics(monkeypatch):
    task = _task("NF_BASE")
    monkeypatch.setattr(
        ae,
        "_frozen_task_map",
        lambda root: {task.run_id: task},
    )
    result = ae._failed(task, "fit", ValueError("boom"), y_true=(1.0,))
    assert result.status == "failed"
    assert result.metrics is None
    assert result.failure is not None
    ae.validate_ablation_result(result)
