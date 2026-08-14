from __future__ import annotations

from collections import Counter

import pytest

import nfpsosc.v2.ablation_runner as ar


def test_new_task_design_has_exact_a010_counts():
    tasks = ar.build_new_ablation_tasks(project_root=".")
    assert len(tasks) == 5250
    assert sum(t.domain == "synthetic" for t in tasks) == 5130
    assert sum(t.domain == "real" for t in tasks) == 120
    assert len({t.run_id for t in tasks}) == 5250


def test_new_task_variant_counts_are_exact():
    tasks = ar.build_new_ablation_tasks(project_root=".")
    counts = Counter((t.domain, t.variant) for t in tasks)

    assert counts[("synthetic", "NF_BASE")] == 270
    for variant in ar.NEW_STOCHASTIC_VARIANTS:
        assert counts[("synthetic", variant)] == 810
    assert counts[("real", "NF_BASE")] == 120
    assert all(t.variant != "PC_NFPSO" for t in tasks)


def test_optimizer_seed_policy_is_exact():
    tasks = ar.build_new_ablation_tasks(project_root=".")
    for task in tasks:
        if task.variant == "NF_BASE":
            assert task.optimizer_seed is None
        else:
            assert task.optimizer_seed in (41001, 41002, 41003)


def test_synthetic_ablation_uses_first_ten_final_dgp_seeds_only():
    tasks = ar.build_new_ablation_tasks(project_root=".")
    observed = {
        t.dgp_seed
        for t in tasks
        if t.domain == "synthetic"
    }
    assert observed == set(range(2001, 2011))


def test_reference_selection_has_exact_810_plus_600_rows():
    rows = ar.select_reference_final_tasks(project_root=".")
    synthetic = [r for r in rows if r.domain == "synthetic"]
    real = [r for r in rows if r.domain == "real"]

    assert len(rows) == 1410
    assert len(synthetic) == 810
    assert len(real) == 600

    assert {r.optimizer_seed for r in synthetic} == {41001, 41002, 41003}
    assert {r.optimizer_seed for r in real} == {
        41001, 41002, 41003, 41004, 41005
    }


def test_new_task_ids_are_namespace_separated_from_external_reference_ids():
    new_ids = {
        t.run_id for t in ar.build_new_ablation_tasks(project_root=".")
    }
    external_ids = {
        r.run_id for r in ar.select_reference_final_tasks(project_root=".")
    }
    assert new_ids.isdisjoint(external_ids)


def test_manifest_roundtrip_without_outcome_execution(tmp_path, monkeypatch):
    tasks = ar.build_new_ablation_tasks(project_root=".")

    refs = tuple(
        ar.ReferenceReuseRow(
            external_run_id=f"{i:064x}",
            domain="synthetic" if i < 810 else "real",
            variant="PC_NFPSO",
            generator="linear_ar" if i < 810 else None,
            dgp_seed=2001 + ((i // 81) % 10) if i < 810 else None,
            noise_level=(0.05, 0.1, 0.2)[(i // 27) % 3] if i < 810 else None,
            source=None if i < 810 else "M3",
            frequency=None if i < 810 else "Yearly",
            series_id=None if i < 810 else f"Y{i-810:04d}",
            optimizer_seed=(41001, 41002, 41003)[i % 3]
            if i < 810
            else (41001, 41002, 41003, 41004, 41005)[(i - 810) % 5],
            result_sha256=f"{(i + 1):064x}",
        )
        for i in range(1410)
    )

    # This test targets deterministic serialization only; reference row semantic
    # coverage is separately validated against the real Final design above.
    monkeypatch.setattr(ar, "validate_reference_reuse_rows", lambda rows: None)

    manifest = ar.AblationManifest(
        schema_version=ar.SCHEMA_VERSION,
        protocol_fingerprint="a" * 64,
        code_commit="b" * 40,
        selected_radius=1.0,
        selected_alpha=0.05,
        synthetic_dgp_seeds=ar.SYNTHETIC_DGP_SEEDS,
        noise_levels=(0.05, 0.1, 0.2),
        ablation_optimizer_seeds=ar.ABLATION_OPTIMIZER_SEEDS,
        real_reference_optimizer_seeds=ar.REAL_REFERENCE_OPTIMIZER_SEEDS,
        variants=tuple(ar.VARIANT_ORDER),
        new_task_count=5250,
        synthetic_new_task_count=5130,
        real_new_task_count=120,
        reference_row_count=1410,
        synthetic_reference_row_count=810,
        real_reference_row_count=600,
        external_final_results_aggregate_sha256="c" * 64,
        tasks=tasks,
        reference_rows=refs,
    )

    ar.validate_ablation_manifest(manifest)
    p, s = ar.write_ablation_manifest(tmp_path, manifest)
    assert p.is_file()
    assert s.is_file()

    observed = ar.load_ablation_manifest(
        tmp_path,
        expected_protocol_fingerprint="a" * 64,
        expected_code_commit="b" * 40,
    )
    assert observed == manifest

    with pytest.raises(ar.AblationManifestError, match="overwrite"):
        ar.write_ablation_manifest(tmp_path, manifest)

def test_external_result_aggregate_is_filename_sorted_not_manifest_order(
    tmp_path,
    monkeypatch,
):
    """Regression for exact compatibility with the frozen 3H closure."""
    from types import SimpleNamespace
    import hashlib

    result_dir = tmp_path / "results"
    result_dir.mkdir()

    tasks = (
        SimpleNamespace(run_id="bbbb"),
        SimpleNamespace(run_id="aaaa"),
    )
    manifest = SimpleNamespace(task_count=8780, tasks=tasks)

    files = {
        "aaaa": b'{"x":1}\n',
        "bbbb": b'{"x":2}\n',
    }
    for run_id, payload in files.items():
        (result_dir / f"{run_id}.json").write_bytes(payload)

    monkeypatch.setattr(
        ar.fe,
        "load_frozen_manifest",
        lambda workspace: manifest,
    )
    monkeypatch.setattr(
        ar.fe,
        "result_path",
        lambda workspace, run_id: result_dir / f"{run_id}.json",
    )

    expected = hashlib.sha256()
    for name in ("aaaa.json", "bbbb.json"):
        expected.update(name.encode("ascii"))
        expected.update(b"\0")
        expected.update((result_dir / name).read_bytes())
        expected.update(b"\0")

    observed = ar.external_results_aggregate_sha256(tmp_path)
    assert observed == expected.hexdigest()
