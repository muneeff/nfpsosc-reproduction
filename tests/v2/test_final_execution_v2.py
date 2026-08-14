from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import numpy as np
import pytest

import nfpsosc.v2.final_execution as fe


def test_frozen_execution_constants_match_manifest_freeze() -> None:
    assert fe.FINAL_MANIFEST_FREEZE_COMMIT == "381288a72157cad2ecaada6618cbdbed18669b5b"
    assert fe.FROZEN_MANIFEST_CODE_COMMIT == "c56504bc8d95ebac9aa6e51c165769903077f348"
    assert fe.FROZEN_PROTOCOL_FINGERPRINT == "545f5be0ec0d14b724ed6dd76189019ab78a72acb39fec0ed53878e2d67700c8"
    assert fe.FROZEN_MANIFEST_SHA256 == "763da2f1e18a1f2572e97f7ece8b17438db0e20c3fe04ac0d05980208f28d5fd"
    assert fe.FROZEN_TOTAL_TASKS == 8780


def test_execution_has_no_radius_or_alpha_override() -> None:
    params = inspect.signature(fe.execute_final_task).parameters
    assert "radius" not in params
    assert "alpha" not in params


def test_metric_bundle_exact_simple_case() -> None:
    train = np.array([1., 2., 4., 7., 11.])
    actual = np.array([13., 17.])
    pred = np.array([12., 20.])
    m = fe.compute_final_metrics(train, actual, pred, seasonal_period=1)
    scale = np.mean(np.abs(np.diff(train)))
    sq = np.mean(np.diff(train) ** 2)
    err = actual - pred
    assert m.mase == pytest.approx(np.mean(np.abs(err)) / scale)
    assert m.mae == pytest.approx(2.0)
    assert m.rmse == pytest.approx(np.sqrt(5.0))
    assert m.smape == pytest.approx(
        200.0 * np.mean(np.abs(err) / (np.abs(actual) + np.abs(pred)))
    )
    assert m.rmsse == pytest.approx(np.sqrt(np.mean(err ** 2) / sq))


def test_smape_zero_zero_term_is_zero() -> None:
    train = np.array([1., 2., 4., 8.])
    actual = np.array([0., 2.])
    pred = np.array([0., 1.])
    m = fe.compute_final_metrics(train, actual, pred, seasonal_period=1)
    expected = 200.0 * ((0.0 + 1.0 / 3.0) / 2.0)
    assert m.smape == pytest.approx(expected)


@pytest.mark.parametrize("m", [0, -1])
def test_metrics_reject_invalid_seasonal_period(m: int) -> None:
    with pytest.raises(fe.FinalExecutionError):
        fe.compute_final_metrics([1., 2., 3.], [4.], [4.], seasonal_period=m)


def test_metrics_reject_zero_scale() -> None:
    with pytest.raises(fe.FinalExecutionError, match="MASE"):
        fe.compute_final_metrics([2., 2., 2., 2.], [2.], [2.], seasonal_period=1)


def test_frozen_manifest_loads_exact_design_without_using_current_head() -> None:
    manifest = fe.load_frozen_manifest(r"outputs\v2\final_external")
    assert manifest.code_commit == fe.FROZEN_MANIFEST_CODE_COMMIT
    assert manifest.protocol_fingerprint == fe.FROZEN_PROTOCOL_FINGERPRINT
    assert manifest.task_count == 8780
    assert manifest.selected_radius == 1.0
    assert manifest.selected_alpha == 0.05


def test_final_freeze_metadata_is_locked_and_hash_verified() -> None:
    freeze = fe._verify_final_freeze_metadata(Path(".").resolve())
    assert freeze["status"] == "LOCKED_PRE_FINAL_OUTCOME_EXECUTION"
    assert freeze["code_commit"] == fe.FROZEN_MANIFEST_CODE_COMMIT
    assert freeze["protocol_fingerprint"] == fe.FROZEN_PROTOCOL_FINGERPRINT
    assert freeze["final_external_manifest"]["sha256"] == fe.FROZEN_MANIFEST_SHA256


def test_pending_is_full_in_fresh_manifest_copy(tmp_path: Path) -> None:
    source = Path("outputs") / "v2" / "final_external"
    (tmp_path / "final_external_manifest.json").write_bytes(
        (source / "final_external_manifest.json").read_bytes()
    )
    (tmp_path / "final_external_manifest.sha256").write_bytes(
        (source / "final_external_manifest.sha256").read_bytes()
    )
    manifest = fe.load_frozen_manifest(tmp_path)
    pending = fe.pending_final_tasks(tmp_path, manifest)
    assert len(pending) == 8780


def test_final_result_roundtrip_with_frozen_task_and_toy_payload(tmp_path: Path) -> None:
    task = fe.load_frozen_manifest(r"outputs\v2\final_external").tasks[0]
    metrics = fe.compute_final_metrics(
        [1., 2., 4., 7., 11., 16.],
        [22., 29.],
        [21., 31.],
        seasonal_period=1,
    )
    result = fe.FinalRunResult(
        task=task,
        status="success",
        metrics=metrics,
        y_true=(22.0, 29.0),
        y_pred=(21.0, 31.0),
        validation_size=27,
        best_objective=0.5,
        n_rules=2,
        best_position_sha256="a" * 64,
        selected_config={},
        selection_failures=(),
        failure=None,
    )
    path = fe.write_final_result(tmp_path, result)
    loaded = fe.load_final_result(path)
    assert loaded == result
    with pytest.raises(fe.FinalExecutionError, match="already exists"):
        fe.write_final_result(tmp_path, result)


def test_tampered_result_task_is_rejected() -> None:
    task = fe.load_frozen_manifest(r"outputs\v2\final_external").tasks[0]
    bad = replace(task, radius=0.75)
    result = fe.FinalRunResult(
        task=bad,
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
            error_type="X",
            message="x",
        ),
    )
    with pytest.raises(fe.FinalExecutionError):
        fe.validate_final_result(result)


def test_real_loader_rejects_non_real_task() -> None:
    task = next(
        t for t in fe.load_frozen_manifest(r"outputs\v2\final_external").tasks
        if t.domain == "synthetic"
    )
    with pytest.raises(fe.FinalExecutionError):
        fe.load_frozen_real_task_series(task, project_root=".")


def test_execute_task_rejects_nonmember_before_data_access() -> None:
    task = fe.load_frozen_manifest(r"outputs\v2\final_external").tasks[0]
    bad = replace(task, run_id="0" * 64)
    with pytest.raises(fe.FinalExecutionError, match="not an exact member"):
        fe.execute_final_task(
            bad,
            project_root=".",
            series_loader=lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("must not be called")
            ),
        )


def test_real_source_snapshot_hashes_verify_without_model_execution() -> None:
    fe._verify_real_source_snapshot(Path(".").resolve())
