from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import nfpsosc.v2.development_runner as dr
from nfpsosc.v2.development_selection import FINAL_DGP_SEEDS


def _first_task() -> dr.DevelopmentRunTask:
    return dr.build_development_tasks()[0]


def _success_result(task: dr.DevelopmentRunTask, mase: float = 1.0) -> dr.DevelopmentRunResult:
    m = dr.seasonal_period(task.generator)
    L = dr.frozen_lag_dimension(m)
    V = dr.pc_nfpso_validation_size(dr.PRETEST_LENGTH, L)
    return dr.DevelopmentRunResult(
        task=task,
        status="success",
        mase=float(mase),
        mase_scale=1.0,
        n_lags=L,
        validation_size=V,
        seasonal_period=m,
        best_objective=0.5,
        n_rules=2,
        failure=None,
    )


def _failed_result(task: dr.DevelopmentRunTask) -> dr.DevelopmentRunResult:
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
        failure=dr.DevelopmentRunFailure("fit", "RuntimeError", "boom"),
    )


def _fake_fit(best_cost: float = 0.25, n_rules: int = 3):
    return SimpleNamespace(
        optimizer_result=SimpleNamespace(best_cost=best_cost),
        final_model=SimpleNamespace(n_rules=n_rules),
    )


def _write_protocol_fixture(root: Path) -> None:
    for i, rel in enumerate(dr.PROTOCOL_INPUT_FILES):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"index": i, "file": rel}) + "\n", encoding="utf-8")


def test_frozen_design_counts_and_dry_run() -> None:
    summary = dr.development_dry_run_summary()
    assert summary.task_count == 28_350
    assert summary.condition_count == 270
    assert summary.generator_count == 9
    assert summary.candidate_pair_count == 35
    assert summary.dgp_seed_count == 10
    assert summary.noise_level_count == 3
    assert summary.optimizer_seed_count == 3
    assert len(summary.first_run_id) == 64
    assert len(summary.last_run_id) == 64
    assert summary.first_run_id != summary.last_run_id


def test_manifest_is_exact_cartesian_product_and_ids_unique() -> None:
    tasks = dr.build_development_tasks()
    assert len(tasks) == 28_350
    assert len({t.run_id for t in tasks}) == len(tasks)
    dr.validate_development_tasks(tasks)


def test_manifest_order_is_deterministic() -> None:
    a = dr.build_development_tasks()
    b = dr.build_development_tasks()
    assert a == b
    assert a[0].generator == dr.ALL_GENERATORS[0]
    assert a[-1].generator == dr.ALL_GENERATORS[-1]


def test_deterministic_run_id_changes_with_any_key_field() -> None:
    t = _first_task()
    baseline = t.run_id
    variants = [
        ("other", t.dgp_seed, t.noise_level, t.optimizer_seed, t.radius, t.alpha),
        (t.generator, t.dgp_seed + 1, t.noise_level, t.optimizer_seed, t.radius, t.alpha),
        (t.generator, t.dgp_seed, 0.10, t.optimizer_seed, t.radius, t.alpha),
        (t.generator, t.dgp_seed, t.noise_level, 31002, t.radius, t.alpha),
        (t.generator, t.dgp_seed, t.noise_level, t.optimizer_seed, 0.35, t.alpha),
        (t.generator, t.dgp_seed, t.noise_level, t.optimizer_seed, t.radius, 1e-5),
    ]
    assert all(dr.deterministic_run_id(*v) != baseline for v in variants)


def test_manifest_rejects_final_seed() -> None:
    tasks = list(dr.build_development_tasks())
    bad = replace(tasks[0], dgp_seed=FINAL_DGP_SEEDS[0])
    bad = replace(
        bad,
        run_id=dr.deterministic_run_id(
            bad.generator, bad.dgp_seed, bad.noise_level, bad.optimizer_seed, bad.radius, bad.alpha
        ),
    )
    tasks[0] = bad
    with pytest.raises(Exception):
        dr.validate_development_tasks(tasks)


def test_validation_size_is_frozen_for_lag_5_and_12() -> None:
    assert dr.pc_nfpso_validation_size(144, 5) == 27
    assert dr.pc_nfpso_validation_size(144, 12) == 26


def test_lag_rule_matches_frozen_period_policy() -> None:
    assert dr.frozen_lag_dimension(1) == 5
    assert dr.frozen_lag_dimension(5) == 5
    assert dr.frozen_lag_dimension(12) == 12
    assert dr.frozen_lag_dimension(24) == 12


def test_mase_scale_uses_only_passed_pretest_values() -> None:
    pre = np.arange(144, dtype=float)
    scale = dr.raw_mase_scale(pre, seasonal_period_value=1)
    assert scale == pytest.approx(1.0)
    test_a = np.zeros(36)
    test_b = np.full(36, 1e12)
    assert dr.raw_mase_scale(pre, seasonal_period_value=1) == scale
    assert not np.array_equal(test_a, test_b)


def test_mase_rejects_degenerate_scale() -> None:
    with pytest.raises(dr.DevelopmentRunnerError):
        dr.raw_mase_scale(np.ones(144), seasonal_period_value=1)


def test_protocol_fingerprint_is_deterministic_and_content_sensitive(tmp_path: Path) -> None:
    _write_protocol_fixture(tmp_path)
    a = dr.protocol_fingerprint(tmp_path)
    b = dr.protocol_fingerprint(tmp_path)
    assert a == b and len(a) == 64
    target = tmp_path / dr.PROTOCOL_INPUT_FILES[-1]
    target.write_text('{"changed":true}\n', encoding="utf-8")
    assert dr.protocol_fingerprint(tmp_path) != a


def test_manifest_round_trip_and_sha256_guard(tmp_path: Path) -> None:
    manifest = dr.build_development_manifest(
        protocol_fingerprint_value="a" * 64,
        code_commit="b" * 40,
    )
    dr.write_development_manifest(tmp_path, manifest)
    loaded = dr.load_development_manifest(
        tmp_path,
        expected_protocol_fingerprint="a" * 64,
        expected_code_commit="b" * 40,
    )
    assert loaded == manifest
    p = tmp_path / "development_manifest.json"
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(dr.DevelopmentRunnerError, match="SHA-256"):
        dr.load_development_manifest(tmp_path)


def test_manifest_refuses_overwrite(tmp_path: Path) -> None:
    manifest = dr.build_development_manifest(
        protocol_fingerprint_value="a" * 64,
        code_commit="b" * 40,
    )
    dr.write_development_manifest(tmp_path, manifest)
    with pytest.raises(dr.DevelopmentRunnerError, match="already exists"):
        dr.write_development_manifest(tmp_path, manifest)


def test_execute_success_uses_frozen_split_lag_validation_and_metrics() -> None:
    task = _first_task()
    captured: dict[str, object] = {}

    def fit(raw_pretest, **kwargs):
        captured["pretest"] = np.asarray(raw_pretest).copy()
        captured.update(kwargs)
        return _fake_fit()

    def forecast(fitted, test_actuals):
        captured["test"] = np.asarray(test_actuals).copy()
        return np.asarray(test_actuals, dtype=float) + 0.5

    result = dr.execute_development_task(task, fit_function=fit, forecast_function=forecast)
    assert result.status == "success"
    assert result.failure is None
    assert len(captured["pretest"]) == 144
    assert len(captured["test"]) == 36
    assert captured["n_lags"] == 5
    assert captured["validation_size"] == 27
    assert captured["radius"] == task.radius
    assert captured["alpha"] == task.alpha
    assert captured["optimizer_seed"] == task.optimizer_seed
    assert np.isfinite(result.mase)
    assert result.best_objective == 0.25
    assert result.n_rules == 3


def test_seasonal_generator_uses_lag_12_and_validation_26() -> None:
    task = next(t for t in dr.build_development_tasks() if t.generator == "noisy_seasonal")
    captured = {}

    def fit(raw_pretest, **kwargs):
        captured.update(kwargs)
        return _fake_fit()

    result = dr.execute_development_task(
        task,
        fit_function=fit,
        forecast_function=lambda fitted, actuals: np.asarray(actuals) + 0.1,
    )
    assert result.status == "success"
    assert captured["n_lags"] == 12
    assert captured["validation_size"] == 26
    assert result.seasonal_period == 12


def test_generation_failure_is_logged_without_mase(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _first_task()
    monkeypatch.setattr(dr, "generate_synthetic_series", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("gen")))
    result = dr.execute_development_task(task)
    assert result.status == "failed"
    assert result.mase is None
    assert result.failure.stage == "generation"
    assert result.failure.error_type == "RuntimeError"


def test_fit_failure_is_logged_without_fallback() -> None:
    task = _first_task()
    def fit(*a, **k):
        raise ValueError("fit failed")
    result = dr.execute_development_task(task, fit_function=fit)
    assert result.status == "failed"
    assert result.mase is None
    assert result.failure.stage == "fit"
    assert result.failure.error_type == "ValueError"


def test_forecast_failure_is_logged_without_fallback() -> None:
    task = _first_task()
    def forecast(*a, **k):
        raise FloatingPointError("forecast failed")
    result = dr.execute_development_task(task, fit_function=lambda *a, **k: _fake_fit(), forecast_function=forecast)
    assert result.status == "failed"
    assert result.mase is None
    assert result.failure.stage == "forecast"


def test_metrics_failure_is_logged_without_mase() -> None:
    task = _first_task()
    result = dr.execute_development_task(
        task,
        fit_function=lambda *a, **k: _fake_fit(best_cost=np.nan),
        forecast_function=lambda fitted, actuals: np.asarray(actuals) + 0.1,
    )
    assert result.status == "failed"
    assert result.failure.stage == "metrics"
    assert result.mase is None


def test_failed_result_cannot_be_converted_to_score() -> None:
    with pytest.raises(dr.DevelopmentRunnerError):
        _failed_result(_first_task()).to_score()


def test_result_atomic_roundtrip_and_duplicate_rejection(tmp_path: Path) -> None:
    result = _success_result(_first_task())
    path = dr.write_development_result(tmp_path, result)
    assert path.is_file()
    assert dr.load_development_result(path) == result
    with pytest.raises(dr.DevelopmentRunnerError, match="already exists"):
        dr.write_development_result(tmp_path, result)


def test_pending_resume_skips_valid_existing_result(tmp_path: Path) -> None:
    all_tasks = dr.build_development_tasks()
    manifest = dr.DevelopmentManifest(
        schema_version=dr.SCHEMA_VERSION,
        protocol_fingerprint="a" * 64,
        code_commit="b" * 40,
        task_count=len(all_tasks),
        condition_count=dr.EXPECTED_CONDITIONS,
        generators=tuple(dr.ALL_GENERATORS),
        dgp_seeds=tuple(dr.DEVELOPMENT_DGP_SEEDS),
        noise_levels=tuple(dr.CONFIRMATORY_NOISE_LEVELS),
        optimizer_seeds=tuple(dr.DEVELOPMENT_OPTIMIZER_SEEDS),
        candidate_pairs=dr.frozen_candidate_grid(),
        tasks=all_tasks,
    )
    dr.write_development_result(tmp_path, _success_result(all_tasks[0]))
    pending = dr.pending_development_tasks(tmp_path, manifest)
    assert len(pending) == dr.EXPECTED_RUNS - 1
    assert all_tasks[0] not in pending


def test_run_pending_respects_max_runs_and_checkpoints(tmp_path: Path) -> None:
    all_tasks = dr.build_development_tasks()
    manifest = dr.DevelopmentManifest(
        schema_version=dr.SCHEMA_VERSION,
        protocol_fingerprint="a" * 64,
        code_commit="b" * 40,
        task_count=len(all_tasks),
        condition_count=dr.EXPECTED_CONDITIONS,
        generators=tuple(dr.ALL_GENERATORS),
        dgp_seeds=tuple(dr.DEVELOPMENT_DGP_SEEDS),
        noise_levels=tuple(dr.CONFIRMATORY_NOISE_LEVELS),
        optimizer_seeds=tuple(dr.DEVELOPMENT_OPTIMIZER_SEEDS),
        candidate_pairs=dr.frozen_candidate_grid(),
        tasks=all_tasks,
    )
    completed = dr.run_pending_development_tasks(
        tmp_path,
        manifest,
        max_runs=2,
        execute_function=lambda task: _success_result(task),
    )
    assert len(completed) == 2
    assert len(dr.collect_development_results(tmp_path, manifest)) == 2
    assert len(dr.pending_development_tasks(tmp_path, manifest)) == dr.EXPECTED_RUNS - 2


def test_selection_gate_blocks_missing_runs() -> None:
    with pytest.raises(dr.DevelopmentRunnerError, match="exactly 28350"):
        dr.selection_from_complete_development_results([_success_result(_first_task())])


def test_selection_gate_blocks_any_failure() -> None:
    tasks = dr.build_development_tasks()
    results = [_success_result(task, mase=1.0) for task in tasks]
    results[123] = _failed_result(tasks[123])
    with pytest.raises(dr.DevelopmentRunnerError, match="blocks radius-alpha selection"):
        dr.selection_from_complete_development_results(results)


def test_selection_gate_accepts_complete_successes_and_applies_tiebreak() -> None:
    tasks = dr.build_development_tasks()
    results = [_success_result(task, mase=2.0) for task in tasks]
    selected = dr.selection_from_complete_development_results(results)
    assert selected.record_count == 28_350
    assert selected.condition_count == 270
    assert selected.generator_count == 9
    assert selected.selected_pair.radius == 1.0
    assert selected.selected_pair.alpha == 0.1


def test_prepare_workspace_requires_clean_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_protocol_fixture(tmp_path)
    monkeypatch.setattr(dr, "assert_clean_git_worktree", lambda root: (_ for _ in ()).throw(dr.DevelopmentRunnerError("dirty")))
    with pytest.raises(dr.DevelopmentRunnerError, match="dirty"):
        dr.prepare_development_workspace(tmp_path / "out", project_root=tmp_path)


def test_prepare_workspace_binds_protocol_and_commit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_protocol_fixture(tmp_path)
    monkeypatch.setattr(dr, "assert_clean_git_worktree", lambda root: None)
    monkeypatch.setattr(dr, "git_head", lambda root: "c" * 40)
    manifest = dr.prepare_development_workspace(tmp_path / "out", project_root=tmp_path)
    assert manifest.code_commit == "c" * 40
    assert manifest.protocol_fingerprint == dr.protocol_fingerprint(tmp_path)
    again = dr.prepare_development_workspace(tmp_path / "out", project_root=tmp_path)
    assert again == manifest


def test_result_validation_rejects_accuracy_on_failed_run() -> None:
    bad = replace(_failed_result(_first_task()), mase=1.0)
    with pytest.raises(dr.DevelopmentRunnerError):
        dr.validate_development_result(bad)
