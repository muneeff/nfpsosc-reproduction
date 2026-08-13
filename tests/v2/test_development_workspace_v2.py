from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

import nfpsosc.v2.development_workspace as ws
from nfpsosc.v2.development_runner import DevelopmentRunTask
from nfpsosc.v2.development_selection import CandidatePair


def frozen_env() -> ws.EnvironmentSnapshot:
    return ws.EnvironmentSnapshot(
        python_version=ws.EXPECTED_PYTHON_VERSION,
        python_implementation="CPython",
        executable="C:/fake/python.exe",
        packages=dict(ws.EXPECTED_PACKAGE_VERSIONS),
        platform="Windows-test",
    )


def test_expected_environment_lock_is_exact():
    assert ws.EXPECTED_PYTHON_VERSION == "3.10.11"
    assert ws.EXPECTED_PACKAGE_VERSIONS == {
        "numpy": "2.2.6",
        "pandas": "2.3.3",
        "scipy": "1.15.3",
        "statsmodels": "0.14.6",
        "scikit-learn": "1.7.2",
        "xgboost": "3.2.0",
    }


def test_assert_frozen_environment_accepts_exact_lock():
    ws.assert_frozen_environment(frozen_env())


@pytest.mark.parametrize(
    "field,value",
    [
        ("python_version", "3.10.12"),
        ("python_implementation", "PyPy"),
    ],
)
def test_environment_rejects_python_mismatch(field, value):
    env = replace(frozen_env(), **{field: value})
    with pytest.raises(ws.DevelopmentWorkspaceError):
        ws.assert_frozen_environment(env)


def test_environment_rejects_package_mismatch():
    env = frozen_env()
    packages = dict(env.packages)
    packages["numpy"] = "9.9.9"
    with pytest.raises(ws.DevelopmentWorkspaceError):
        ws.assert_frozen_environment(replace(env, packages=packages))


def test_canonical_json_is_deterministic():
    a = ws._canonical_json_bytes({"b": 2, "a": 1})
    b = ws._canonical_json_bytes({"a": 1, "b": 2})
    assert a == b
    assert a.endswith(b"\n")


def test_sha256_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert ws.sha256_file(p) == hashlib.sha256(b"abc").hexdigest()


def make_manifest():
    pairs = (
        CandidatePair(0.25, 1e-6),
        CandidatePair(1.0, 0.1),
    )
    tasks = tuple(
        DevelopmentRunTask(f"{i:064x}", "g", 1001, 0.05, 31001, 0.25, 1e-6)
        for i in range(ws.EXPECTED_RUNS)
    )
    return ws.DevelopmentManifest(
        schema_version=ws.DEVELOPMENT_RUNNER_SCHEMA_VERSION,
        protocol_fingerprint="a" * 64,
        code_commit="b" * 40,
        task_count=ws.EXPECTED_RUNS,
        condition_count=ws.EXPECTED_CONDITIONS,
        generators=("g",),
        dgp_seeds=(1001,),
        noise_levels=(0.05,),
        optimizer_seeds=(31001,),
        candidate_pairs=pairs,
        tasks=tasks,
    )


def write_fake_manifest_files(out: Path, digest_source: bytes = b"manifest\n"):
    out.mkdir(parents=True, exist_ok=True)
    (out / ws.DEVELOPMENT_MANIFEST_FILENAME).write_bytes(digest_source)
    digest = hashlib.sha256(digest_source).hexdigest()
    (out / ws.DEVELOPMENT_MANIFEST_HASH_FILENAME).write_text(digest + "\n", encoding="ascii")
    return digest


def install_common_monkeypatches(monkeypatch, manifest):
    monkeypatch.setattr(ws, "assert_clean_git_worktree", lambda root: None)
    monkeypatch.setattr(ws, "capture_environment", frozen_env)
    monkeypatch.setattr(ws, "git_head", lambda root: manifest.code_commit)
    monkeypatch.setattr(
        ws, "protocol_fingerprint", lambda root: manifest.protocol_fingerprint
    )
    monkeypatch.setattr(
        ws,
        "prepare_development_workspace",
        lambda out, project_root: manifest,
    )
    monkeypatch.setattr(
        ws,
        "load_development_manifest",
        lambda out, expected_protocol_fingerprint=None, expected_code_commit=None: manifest,
    )
    monkeypatch.setattr(ws, "collect_development_results", lambda out, m: ())
    monkeypatch.setattr(ws, "pending_development_tasks", lambda out, m: m.tasks)


def test_manifest_digest_sidecar_verification(tmp_path):
    digest = write_fake_manifest_files(tmp_path)
    assert ws._manifest_digest_from_sidecar(tmp_path) == digest
    (tmp_path / ws.DEVELOPMENT_MANIFEST_FILENAME).write_bytes(b"tampered")
    with pytest.raises(ws.DevelopmentWorkspaceError):
        ws._manifest_digest_from_sidecar(tmp_path)


def test_freeze_pre_outcome_workspace(monkeypatch, tmp_path):
    manifest = make_manifest()
    install_common_monkeypatches(monkeypatch, manifest)
    manifest_digest = write_fake_manifest_files(tmp_path)

    freeze = ws.freeze_development_workspace(tmp_path, project_root=".")
    assert freeze.code_commit == manifest.code_commit
    assert freeze.protocol_fingerprint == manifest.protocol_fingerprint
    assert freeze.task_count == ws.EXPECTED_RUNS
    assert freeze.condition_count == ws.EXPECTED_CONDITIONS
    assert freeze.development_manifest_sha256 == manifest_digest
    assert freeze.first_run_id == manifest.tasks[0].run_id
    assert freeze.last_run_id == manifest.tasks[-1].run_id
    assert (tmp_path / ws.WORKSPACE_FREEZE_FILENAME).is_file()
    assert (tmp_path / ws.WORKSPACE_FREEZE_HASH_FILENAME).is_file()


def test_freeze_refuses_existing_result(monkeypatch, tmp_path):
    manifest = make_manifest()
    install_common_monkeypatches(monkeypatch, manifest)
    write_fake_manifest_files(tmp_path)
    monkeypatch.setattr(
        ws,
        "collect_development_results",
        lambda out, m: (object(),),
    )
    with pytest.raises(ws.DevelopmentWorkspaceError, match="results already exist"):
        ws.freeze_development_workspace(tmp_path, project_root=".")


def test_load_freeze_detects_tampering(monkeypatch, tmp_path):
    manifest = make_manifest()
    install_common_monkeypatches(monkeypatch, manifest)
    write_fake_manifest_files(tmp_path)
    ws.freeze_development_workspace(tmp_path, project_root=".")
    p = tmp_path / ws.WORKSPACE_FREEZE_FILENAME
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(ws.DevelopmentWorkspaceError, match="SHA-256 mismatch"):
        ws.load_workspace_freeze(tmp_path)


def test_verify_workspace_zero_outcome(monkeypatch, tmp_path):
    manifest = make_manifest()
    install_common_monkeypatches(monkeypatch, manifest)
    digest = write_fake_manifest_files(tmp_path)
    ws.freeze_development_workspace(tmp_path, project_root=".")

    verification = ws.verify_development_workspace(
        tmp_path, project_root=".", require_no_results=True
    )
    assert verification.manifest_sha256 == digest
    assert verification.completed_count == 0
    assert verification.failed_count == 0
    assert verification.pending_count == ws.EXPECTED_RUNS
    assert verification.ready_for_first_run is True


def test_verify_rejects_current_head_change(monkeypatch, tmp_path):
    manifest = make_manifest()
    install_common_monkeypatches(monkeypatch, manifest)
    write_fake_manifest_files(tmp_path)
    ws.freeze_development_workspace(tmp_path, project_root=".")
    monkeypatch.setattr(ws, "git_head", lambda root: "c" * 40)
    with pytest.raises(ws.DevelopmentWorkspaceError, match="bound to code commit"):
        ws.verify_development_workspace(tmp_path, project_root=".")


def test_verify_rejects_protocol_change(monkeypatch, tmp_path):
    manifest = make_manifest()
    install_common_monkeypatches(monkeypatch, manifest)
    write_fake_manifest_files(tmp_path)
    ws.freeze_development_workspace(tmp_path, project_root=".")
    monkeypatch.setattr(ws, "protocol_fingerprint", lambda root: "d" * 64)
    with pytest.raises(ws.DevelopmentWorkspaceError, match="protocol fingerprint"):
        ws.verify_development_workspace(tmp_path, project_root=".")


def test_verify_rejects_environment_change(monkeypatch, tmp_path):
    manifest = make_manifest()
    install_common_monkeypatches(monkeypatch, manifest)
    write_fake_manifest_files(tmp_path)
    ws.freeze_development_workspace(tmp_path, project_root=".")
    bad = replace(frozen_env(), packages={**frozen_env().packages, "numpy": "2.2.7"})
    monkeypatch.setattr(ws, "capture_environment", lambda: bad)
    with pytest.raises(ws.DevelopmentWorkspaceError):
        ws.verify_development_workspace(tmp_path, project_root=".")


def test_verify_rejects_orphan_result_file(monkeypatch, tmp_path):
    manifest = make_manifest()
    install_common_monkeypatches(monkeypatch, manifest)
    write_fake_manifest_files(tmp_path)
    ws.freeze_development_workspace(tmp_path, project_root=".")
    result_dir = tmp_path / ws.RESULTS_DIRECTORY_NAME
    result_dir.mkdir()
    (result_dir / ("f" * 64 + ".json")).write_text("{}", encoding="utf-8")
    with pytest.raises(ws.DevelopmentWorkspaceError, match="Orphan"):
        ws.verify_development_workspace(tmp_path, project_root=".")


def test_first_run_gate_rejects_completed_result(monkeypatch, tmp_path):
    manifest = make_manifest()
    install_common_monkeypatches(monkeypatch, manifest)
    write_fake_manifest_files(tmp_path)
    ws.freeze_development_workspace(tmp_path, project_root=".")

    class Result:
        successful = True

    monkeypatch.setattr(ws, "collect_development_results", lambda out, m: (Result(),))
    monkeypatch.setattr(
        ws, "pending_development_tasks", lambda out, m: m.tasks[1:]
    )
    with pytest.raises(ws.DevelopmentWorkspaceError, match="zero pre-existing"):
        ws.assert_ready_for_first_development_run(tmp_path, project_root=".")


def test_partial_freeze_is_rejected(monkeypatch, tmp_path):
    manifest = make_manifest()
    install_common_monkeypatches(monkeypatch, manifest)
    write_fake_manifest_files(tmp_path)
    (tmp_path / ws.WORKSPACE_FREEZE_FILENAME).write_text("{}", encoding="utf-8")
    with pytest.raises(ws.DevelopmentWorkspaceError, match="Partial"):
        ws.freeze_development_workspace(tmp_path, project_root=".")
