from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import pytest

import nfpsosc.v2.final_runner as fr
from nfpsosc.v2.development_selection import (
    DEVELOPMENT_DGP_SEEDS,
    DEVELOPMENT_OPTIMIZER_SEEDS,
    FINAL_DGP_SEEDS,
    FINAL_OPTIMIZER_SEEDS,
    LEGACY_V1_DGP_SEEDS,
)


def test_manifest_stage_cannot_execute_final_outcomes() -> None:
    assert fr.FINAL_OUTCOME_EXECUTION_ENABLED is False
    assert not hasattr(fr, "execute_final_task")
    assert not hasattr(fr, "run_pending_final_tasks")


def test_no_radius_or_alpha_override_exists_on_public_build_prepare_interfaces() -> None:
    for fn in (
        fr.build_final_tasks,
        fr.final_dry_run_summary,
        fr.prepare_final_workspace,
    ):
        params = inspect.signature(fn).parameters
        assert "radius" not in params
        assert "alpha" not in params


def test_protocol_fingerprint_includes_selection_freeze_a009_and_all_amendments() -> None:
    inputs = set(fr.PROTOCOL_INPUT_FILES)
    assert "configs/v2/development_selection_freeze_v2.json" in inputs
    assert "configs/v2/amendment_009_final_aggregation_metrics.json" in inputs
    for i in range(1, 10):
        prefix = f"configs/v2/amendment_{i:03d}_"
        assert any(path.startswith(prefix) and not path.endswith("sha256.json") for path in inputs)


def test_dry_run_exact_final_design_counts() -> None:
    s = fr.final_dry_run_summary(project_root=".")
    assert s.task_count == 8_780
    assert s.synthetic_task_count == 7_140
    assert s.real_task_count == 1_640
    assert s.synthetic_pc_task_count == 2_700
    assert s.synthetic_baseline_task_count == 4_440
    assert s.real_pc_task_count == 600
    assert s.real_baseline_task_count == 1_040
    assert s.real_final_series_count == 120
    assert s.generator_count == 9
    assert s.baseline_count == 9
    assert s.final_dgp_seed_count == 20
    assert s.final_optimizer_seed_count == 5
    assert s.selected_radius == 1.0
    assert s.selected_alpha == 0.05
    assert len(s.first_run_id) == 64
    assert len(s.last_run_id) == 64
    assert s.first_run_id != s.last_run_id


def test_final_task_ids_are_unique_and_design_validates() -> None:
    tasks = fr.build_final_tasks(project_root=".")
    assert len(tasks) == fr.EXPECTED_TOTAL_TASKS
    assert len({t.run_id for t in tasks}) == len(tasks)
    fr.validate_final_tasks(tasks)


def test_final_seed_firewall_excludes_development_and_legacy_sets() -> None:
    tasks = fr.build_final_tasks(project_root=".")
    synthetic = [t for t in tasks if t.domain == "synthetic"]
    observed_dgp = {t.dgp_seed for t in synthetic}
    assert observed_dgp == set(FINAL_DGP_SEEDS)
    assert not observed_dgp.intersection(DEVELOPMENT_DGP_SEEDS)
    assert not observed_dgp.intersection(LEGACY_V1_DGP_SEEDS)

    pc = [t for t in tasks if t.method == fr.PC_METHOD]
    observed_opt = {t.optimizer_seed for t in pc}
    assert observed_opt == set(FINAL_OPTIMIZER_SEEDS)
    assert not observed_opt.intersection(DEVELOPMENT_OPTIMIZER_SEEDS)


def test_all_pc_tasks_use_only_frozen_development_selected_pair() -> None:
    pc = [
        t for t in fr.build_final_tasks(project_root=".")
        if t.method == fr.PC_METHOD
    ]
    assert len(pc) == 3_300
    assert {t.radius for t in pc} == {1.0}
    assert {t.alpha for t in pc} == {0.05}


def test_baselines_have_no_pc_optimizer_seed_or_pc_hyperparameters() -> None:
    baselines = [
        t for t in fr.build_final_tasks(project_root=".")
        if t.method != fr.PC_METHOD
    ]
    assert baselines
    assert all(t.optimizer_seed is None for t in baselines)
    assert all(t.radius is None for t in baselines)
    assert all(t.alpha is None for t in baselines)


def test_seasonal_naive_rows_exist_only_for_m_greater_than_one() -> None:
    seasonal = [
        t for t in fr.build_final_tasks(project_root=".")
        if t.method == "seasonal_naive"
    ]
    assert seasonal
    assert all(t.seasonal_period > 1 for t in seasonal)
    assert sum(t.domain == "synthetic" for t in seasonal) == 120
    assert sum(t.domain == "real" for t in seasonal) == 80


def test_real_tasks_cover_exactly_120_final_series_and_no_synthetic_identity() -> None:
    real = [
        t for t in fr.build_final_tasks(project_root=".")
        if t.domain == "real"
    ]
    identities = {(t.source, t.frequency, t.series_id) for t in real}
    assert len(identities) == 120
    assert all(t.generator is None for t in real)
    assert all(t.dgp_seed is None for t in real)
    assert all(t.noise_level is None for t in real)


def test_synthetic_tasks_cover_exactly_nine_generators_and_final_conditions() -> None:
    synthetic = [
        t for t in fr.build_final_tasks(project_root=".")
        if t.domain == "synthetic"
    ]
    assert len({t.generator for t in synthetic}) == 9
    assert {t.dgp_seed for t in synthetic} == set(FINAL_DGP_SEEDS)
    assert {t.noise_level for t in synthetic} == {0.05, 0.10, 0.20}
    assert all(t.source is None for t in synthetic)
    assert all(t.frequency is None for t in synthetic)
    assert all(t.series_id is None for t in synthetic)


def test_task_tampering_is_rejected() -> None:
    tasks = list(fr.build_final_tasks(project_root="."))
    pc_index = next(i for i, t in enumerate(tasks) if t.method == fr.PC_METHOD)
    tasks[pc_index] = replace(tasks[pc_index], radius=0.75)
    with pytest.raises(fr.FinalManifestError):
        fr.validate_final_tasks(tasks)


def test_manifest_roundtrip_sha_guard_and_overwrite_refusal(tmp_path: Path) -> None:
    manifest = fr.build_final_manifest(
        project_root=".",
        protocol_fingerprint_value="a" * 64,
        code_commit="b" * 40,
    )
    manifest_path, hash_path = fr.write_final_manifest(tmp_path, manifest)
    assert manifest_path.is_file()
    assert hash_path.is_file()

    payload = manifest_path.read_bytes()
    expected = hash_path.read_text(encoding="ascii").strip()
    assert hashlib.sha256(payload).hexdigest() == expected

    loaded = fr.load_final_manifest(
        tmp_path,
        expected_protocol_fingerprint="a" * 64,
        expected_code_commit="b" * 40,
    )
    assert loaded == manifest

    with pytest.raises(fr.FinalManifestError, match="already exists"):
        fr.write_final_manifest(tmp_path, manifest)

    manifest_path.write_bytes(payload + b" ")
    with pytest.raises(fr.FinalManifestError, match="SHA-256"):
        fr.load_final_manifest(tmp_path)


def test_manifest_payload_marks_execution_disabled(tmp_path: Path) -> None:
    manifest = fr.build_final_manifest(
        project_root=".",
        protocol_fingerprint_value="c" * 64,
        code_commit="d" * 40,
    )
    path, _ = fr.write_final_manifest(tmp_path, manifest)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["stage"] == "manifest_only_pre_final_outcomes"
    assert data["final_outcome_execution_enabled"] is False
    assert data["selected_radius"] == 1.0
    assert data["selected_alpha"] == 0.05
    assert data["task_count"] == 8_780
