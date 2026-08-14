from __future__ import annotations

"""Manifest-only design layer for the frozen V2 internal ablation study.

This module constructs and validates the *new* ablation task design and binds
the PC_NFPSO reference rows to the already-closed external Final corpus.

It does not fit, forecast, optimize, compute metrics, or execute an ablation
task. Production execution must be implemented and frozen separately after the
manifest itself has been frozen.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Literal, Sequence

from . import final_execution as fe
from . import final_runner as fr
from .ablation import VARIANT_ORDER
from .development_selection import CONFIRMATORY_NOISE_LEVELS
from .synthetic import ALL_GENERATORS, seasonal_period


SCHEMA_VERSION = "v2-internal-ablation-manifest-1"
TASK_NAMESPACE = "v2-internal-ablation-new-task-1"
MANIFEST_FILENAME = "ablation_manifest.json"
MANIFEST_SHA_FILENAME = "ablation_manifest_sha256.json"

EXPECTED_RADIUS = 1.0
EXPECTED_ALPHA = 0.05

SYNTHETIC_DGP_SEEDS: tuple[int, ...] = tuple(range(2001, 2011))
ABLATION_OPTIMIZER_SEEDS: tuple[int, ...] = (41001, 41002, 41003)
REAL_REFERENCE_OPTIMIZER_SEEDS: tuple[int, ...] = (
    41001,
    41002,
    41003,
    41004,
    41005,
)

NEW_STOCHASTIC_VARIANTS: tuple[str, ...] = (
    "NFPSO",
    "P_NFPSO",
    "C_NFPSO",
    "PC_NO_SENSITIVITY",
    "PC_NO_COVERAGE",
    "PC_VALIDATION_ONLY",
)

REFERENCE_VARIANT = "PC_NFPSO"

EXPECTED_SYNTHETIC_CONDITIONS = 270
EXPECTED_SYNTHETIC_NF_BASE_TASKS = 270
EXPECTED_SYNTHETIC_STOCHASTIC_TASKS = 4860
EXPECTED_SYNTHETIC_NEW_TASKS = 5130
EXPECTED_REAL_NEW_TASKS = 120
EXPECTED_NEW_TASKS = 5250

EXPECTED_SYNTHETIC_REFERENCE_ROWS = 810
EXPECTED_REAL_REFERENCE_ROWS = 600
EXPECTED_REFERENCE_ROWS = 1410
EXPECTED_EXTERNAL_TOTAL_RESULTS = 8780

PROTOCOL_INPUT_FILES: tuple[str, ...] = (
    "configs/v2/ablation_v2.json",
    "configs/v2/amendment_010_ablation_execution_provenance.json",
    "configs/v2/amendment_010_sha256.json",
    "configs/v2/development_selection_freeze_v2.json",
    "configs/v2/development_selection_freeze_sha256_v2.json",
    "configs/v2/synthetic_split_v2.json",
    "configs/v2/canonical_synthetic_v2.json",
    "configs/v2/real_series_manifest_v2.json",
    "configs/v2/metrics_v2.json",
)


class AblationManifestError(RuntimeError):
    """Raised when the frozen ablation design/provenance contract is violated."""


@dataclass(frozen=True)
class AblationRunTask:
    run_id: str
    domain: Literal["synthetic", "real"]
    variant: str
    generator: str | None
    dgp_seed: int | None
    noise_level: float | None
    source: str | None
    frequency: str | None
    series_id: str | None
    optimizer_seed: int | None
    seasonal_period: int
    n_lags: int
    radius: float
    alpha: float


@dataclass(frozen=True)
class ReferenceReuseRow:
    external_run_id: str
    domain: Literal["synthetic", "real"]
    variant: str
    generator: str | None
    dgp_seed: int | None
    noise_level: float | None
    source: str | None
    frequency: str | None
    series_id: str | None
    optimizer_seed: int
    result_sha256: str


@dataclass(frozen=True)
class AblationManifest:
    schema_version: str
    protocol_fingerprint: str
    code_commit: str
    selected_radius: float
    selected_alpha: float
    synthetic_dgp_seeds: tuple[int, ...]
    noise_levels: tuple[float, ...]
    ablation_optimizer_seeds: tuple[int, ...]
    real_reference_optimizer_seeds: tuple[int, ...]
    variants: tuple[str, ...]
    new_task_count: int
    synthetic_new_task_count: int
    real_new_task_count: int
    reference_row_count: int
    synthetic_reference_row_count: int
    real_reference_row_count: int
    external_final_results_aggregate_sha256: str
    tasks: tuple[AblationRunTask, ...]
    reference_rows: tuple[ReferenceReuseRow, ...]


def _canonical_float(value: float | None) -> str | None:
    if value is None:
        return None
    return format(float(value), ".17g")


def _canonical_task_payload(
    *,
    domain: str,
    variant: str,
    generator: str | None,
    dgp_seed: int | None,
    noise_level: float | None,
    source: str | None,
    frequency: str | None,
    series_id: str | None,
    optimizer_seed: int | None,
    seasonal_period_value: int,
    n_lags: int,
    radius: float,
    alpha: float,
) -> dict[str, Any]:
    return {
        "namespace": TASK_NAMESPACE,
        "domain": str(domain),
        "variant": str(variant),
        "generator": generator,
        "dgp_seed": None if dgp_seed is None else int(dgp_seed),
        "noise_level": _canonical_float(noise_level),
        "source": source,
        "frequency": frequency,
        "series_id": series_id,
        "optimizer_seed": None if optimizer_seed is None else int(optimizer_seed),
        "seasonal_period": int(seasonal_period_value),
        "n_lags": int(n_lags),
        "radius": _canonical_float(radius),
        "alpha": _canonical_float(alpha),
    }


def deterministic_ablation_run_id(**kwargs: Any) -> str:
    payload = _canonical_task_payload(**kwargs)
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _make_task(
    *,
    domain: Literal["synthetic", "real"],
    variant: str,
    generator: str | None = None,
    dgp_seed: int | None = None,
    noise_level: float | None = None,
    source: str | None = None,
    frequency: str | None = None,
    series_id: str | None = None,
    optimizer_seed: int | None = None,
    seasonal_period_value: int,
    n_lags: int,
) -> AblationRunTask:
    run_id = deterministic_ablation_run_id(
        domain=domain,
        variant=variant,
        generator=generator,
        dgp_seed=dgp_seed,
        noise_level=noise_level,
        source=source,
        frequency=frequency,
        series_id=series_id,
        optimizer_seed=optimizer_seed,
        seasonal_period_value=seasonal_period_value,
        n_lags=n_lags,
        radius=EXPECTED_RADIUS,
        alpha=EXPECTED_ALPHA,
    )
    return AblationRunTask(
        run_id=run_id,
        domain=domain,
        variant=str(variant),
        generator=generator,
        dgp_seed=dgp_seed,
        noise_level=noise_level,
        source=source,
        frequency=frequency,
        series_id=series_id,
        optimizer_seed=optimizer_seed,
        seasonal_period=int(seasonal_period_value),
        n_lags=int(n_lags),
        radius=EXPECTED_RADIUS,
        alpha=EXPECTED_ALPHA,
    )


def _final_pc_tasks(
    *, project_root: str | os.PathLike[str] = "."
) -> tuple[fr.FinalRunTask, ...]:
    rows = tuple(
        task
        for task in fr.build_final_tasks(project_root=project_root)
        if task.method == fr.PC_METHOD
    )
    expected = 2700 + 600
    if len(rows) != expected:
        raise AblationManifestError(
            f"Unexpected Final PC-NFPSO task count: {len(rows)} != {expected}."
        )
    return rows


def _real_series_prototypes(
    *, project_root: str | os.PathLike[str] = "."
) -> tuple[fr.FinalRunTask, ...]:
    pc_rows = _final_pc_tasks(project_root=project_root)
    by_key: dict[tuple[str, str, str], fr.FinalRunTask] = {}
    for task in pc_rows:
        if task.domain != "real":
            continue
        if not task.source or not task.frequency or not task.series_id:
            raise AblationManifestError("Final real PC reference identity is incomplete.")
        key = (task.source, task.frequency, task.series_id)
        by_key.setdefault(key, task)

    rows = tuple(by_key[key] for key in sorted(by_key))
    if len(rows) != EXPECTED_REAL_NEW_TASKS:
        raise AblationManifestError(
            f"Expected 120 unique real series, found {len(rows)}."
        )
    return rows


def build_new_ablation_tasks(
    *, project_root: str | os.PathLike[str] = "."
) -> tuple[AblationRunTask, ...]:
    tasks: list[AblationRunTask] = []

    for generator in ALL_GENERATORS:
        m = seasonal_period(generator)
        L = fr.frozen_lag_dimension(m)
        for dgp_seed in SYNTHETIC_DGP_SEEDS:
            for noise_level in CONFIRMATORY_NOISE_LEVELS:
                tasks.append(
                    _make_task(
                        domain="synthetic",
                        variant="NF_BASE",
                        generator=generator,
                        dgp_seed=int(dgp_seed),
                        noise_level=float(noise_level),
                        optimizer_seed=None,
                        seasonal_period_value=m,
                        n_lags=L,
                    )
                )
                for variant in NEW_STOCHASTIC_VARIANTS:
                    for optimizer_seed in ABLATION_OPTIMIZER_SEEDS:
                        tasks.append(
                            _make_task(
                                domain="synthetic",
                                variant=variant,
                                generator=generator,
                                dgp_seed=int(dgp_seed),
                                noise_level=float(noise_level),
                                optimizer_seed=int(optimizer_seed),
                                seasonal_period_value=m,
                                n_lags=L,
                            )
                        )

    for proto in _real_series_prototypes(project_root=project_root):
        tasks.append(
            _make_task(
                domain="real",
                variant="NF_BASE",
                source=proto.source,
                frequency=proto.frequency,
                series_id=proto.series_id,
                optimizer_seed=None,
                seasonal_period_value=proto.seasonal_period,
                n_lags=proto.n_lags,
            )
        )

    rows = tuple(tasks)
    validate_new_ablation_tasks(rows)
    return rows


def validate_new_ablation_tasks(tasks: Sequence[AblationRunTask]) -> None:
    rows = tuple(tasks)
    if len(rows) != EXPECTED_NEW_TASKS:
        raise AblationManifestError(
            f"Ablation design requires {EXPECTED_NEW_TASKS} new tasks; got {len(rows)}."
        )

    ids = [r.run_id for r in rows]
    if len(ids) != len(set(ids)):
        raise AblationManifestError("Duplicate ablation run IDs detected.")

    synthetic = tuple(r for r in rows if r.domain == "synthetic")
    real = tuple(r for r in rows if r.domain == "real")
    if len(synthetic) != EXPECTED_SYNTHETIC_NEW_TASKS:
        raise AblationManifestError("Synthetic new-task count mismatch.")
    if len(real) != EXPECTED_REAL_NEW_TASKS:
        raise AblationManifestError("Real new-task count mismatch.")

    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        counts[(row.domain, row.variant)] = counts.get(
            (row.domain, row.variant), 0
        ) + 1

        if row.radius != EXPECTED_RADIUS or row.alpha != EXPECTED_ALPHA:
            raise AblationManifestError("Selected radius/alpha drift detected.")
        if row.seasonal_period < 1:
            raise AblationManifestError("Invalid seasonal period.")
        if row.n_lags != fr.frozen_lag_dimension(row.seasonal_period):
            raise AblationManifestError("Frozen lag-policy mismatch.")
        if row.variant == REFERENCE_VARIANT:
            raise AblationManifestError(
                "PC_NFPSO reference must be reused, not scheduled as a new task."
            )

        if row.variant == "NF_BASE":
            if row.optimizer_seed is not None:
                raise AblationManifestError("NF_BASE cannot carry an optimizer seed.")
        elif row.variant in NEW_STOCHASTIC_VARIANTS:
            if row.optimizer_seed not in ABLATION_OPTIMIZER_SEEDS:
                raise AblationManifestError(
                    "Stochastic ablation task uses a non-frozen optimizer seed."
                )
        else:
            raise AblationManifestError(f"Unexpected ablation variant: {row.variant!r}")

        expected_id = deterministic_ablation_run_id(
            domain=row.domain,
            variant=row.variant,
            generator=row.generator,
            dgp_seed=row.dgp_seed,
            noise_level=row.noise_level,
            source=row.source,
            frequency=row.frequency,
            series_id=row.series_id,
            optimizer_seed=row.optimizer_seed,
            seasonal_period_value=row.seasonal_period,
            n_lags=row.n_lags,
            radius=row.radius,
            alpha=row.alpha,
        )
        if row.run_id != expected_id:
            raise AblationManifestError("Ablation task run ID mismatch.")

        if row.domain == "synthetic":
            if row.generator not in ALL_GENERATORS:
                raise AblationManifestError("Unknown synthetic process family.")
            if row.dgp_seed not in SYNTHETIC_DGP_SEEDS:
                raise AblationManifestError("Synthetic ablation DGP seed mismatch.")
            if row.noise_level not in CONFIRMATORY_NOISE_LEVELS:
                raise AblationManifestError("Synthetic noise level mismatch.")
            if any(v is not None for v in (row.source, row.frequency, row.series_id)):
                raise AblationManifestError(
                    "Synthetic ablation task contains real-series identity."
                )
        elif row.domain == "real":
            if row.variant != "NF_BASE":
                raise AblationManifestError(
                    "Only NF_BASE is newly executed on the real ablation anchor."
                )
            if any(v is not None for v in (row.generator, row.dgp_seed, row.noise_level)):
                raise AblationManifestError(
                    "Real ablation task contains synthetic identity."
                )
            if not row.source or not row.frequency or not row.series_id:
                raise AblationManifestError("Real ablation identity is incomplete.")
        else:
            raise AblationManifestError(f"Unsupported ablation domain: {row.domain!r}")

    if counts.get(("synthetic", "NF_BASE"), 0) != EXPECTED_SYNTHETIC_NF_BASE_TASKS:
        raise AblationManifestError("Synthetic NF_BASE task count mismatch.")
    for variant in NEW_STOCHASTIC_VARIANTS:
        if counts.get(("synthetic", variant), 0) != 810:
            raise AblationManifestError(
                f"Synthetic {variant} task count must be exactly 810."
            )
    if counts.get(("real", "NF_BASE"), 0) != EXPECTED_REAL_NEW_TASKS:
        raise AblationManifestError("Real NF_BASE task count mismatch.")


def select_reference_final_tasks(
    *, project_root: str | os.PathLike[str] = "."
) -> tuple[fr.FinalRunTask, ...]:
    rows = []
    for task in _final_pc_tasks(project_root=project_root):
        if task.domain == "synthetic":
            if (
                task.dgp_seed in SYNTHETIC_DGP_SEEDS
                and task.noise_level in CONFIRMATORY_NOISE_LEVELS
                and task.optimizer_seed in ABLATION_OPTIMIZER_SEEDS
            ):
                rows.append(task)
        elif task.domain == "real":
            if task.optimizer_seed in REAL_REFERENCE_OPTIMIZER_SEEDS:
                rows.append(task)

    selected = tuple(rows)
    synthetic = tuple(r for r in selected if r.domain == "synthetic")
    real = tuple(r for r in selected if r.domain == "real")

    if len(synthetic) != EXPECTED_SYNTHETIC_REFERENCE_ROWS:
        raise AblationManifestError(
            f"Expected 810 synthetic reference rows, found {len(synthetic)}."
        )
    if len(real) != EXPECTED_REAL_REFERENCE_ROWS:
        raise AblationManifestError(
            f"Expected 600 real reference rows, found {len(real)}."
        )
    if len({r.run_id for r in selected}) != EXPECTED_REFERENCE_ROWS:
        raise AblationManifestError("Duplicate external reference run IDs detected.")

    return selected


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def external_results_aggregate_sha256(
    external_workspace: str | os.PathLike[str],
) -> str:
    """Reproduce the frozen 3H Final-result aggregate byte-for-byte.

    The 3H closure sorts result paths lexicographically by filename before
    concatenating name + NUL + bytes + NUL into the SHA-256 stream.
    Manifest task order is deliberately irrelevant here.
    """
    manifest = fe.load_frozen_manifest(external_workspace)
    if manifest.task_count != EXPECTED_EXTERNAL_TOTAL_RESULTS:
        raise AblationManifestError("External Final manifest task count mismatch.")

    paths = sorted(
        fe.result_path(external_workspace, task.run_id)
        for task in manifest.tasks
    )

    h = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise AblationManifestError(
                f"Missing closed external Final result: {path.name}"
            )
        h.update(path.name.encode("ascii"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def build_reference_reuse_rows(
    *,
    external_workspace: str | os.PathLike[str],
    project_root: str | os.PathLike[str] = ".",
) -> tuple[ReferenceReuseRow, ...]:
    rows: list[ReferenceReuseRow] = []

    for task in select_reference_final_tasks(project_root=project_root):
        path = fe.result_path(external_workspace, task.run_id)
        if not path.is_file():
            raise AblationManifestError(
                f"Missing external reference result: {path.name}"
            )
        result = fe.load_final_result(path)
        if result.task != task:
            raise AblationManifestError(
                f"External reference task mismatch: {path.name}"
            )
        if not result.successful:
            raise AblationManifestError(
                f"External reference result is not successful: {path.name}"
            )
        if task.optimizer_seed is None:
            raise AblationManifestError("PC_NFPSO reference cannot lack optimizer seed.")

        rows.append(
            ReferenceReuseRow(
                external_run_id=task.run_id,
                domain=task.domain,
                variant=REFERENCE_VARIANT,
                generator=task.generator,
                dgp_seed=task.dgp_seed,
                noise_level=task.noise_level,
                source=task.source,
                frequency=task.frequency,
                series_id=task.series_id,
                optimizer_seed=int(task.optimizer_seed),
                result_sha256=_sha256_file(path),
            )
        )

    result_rows = tuple(rows)
    validate_reference_reuse_rows(result_rows)
    return result_rows


def validate_reference_reuse_rows(rows: Sequence[ReferenceReuseRow]) -> None:
    values = tuple(rows)
    if len(values) != EXPECTED_REFERENCE_ROWS:
        raise AblationManifestError("Reference-row count mismatch.")
    if len({r.external_run_id for r in values}) != len(values):
        raise AblationManifestError("Duplicate external reference run IDs.")

    synthetic = tuple(r for r in values if r.domain == "synthetic")
    real = tuple(r for r in values if r.domain == "real")
    if len(synthetic) != EXPECTED_SYNTHETIC_REFERENCE_ROWS:
        raise AblationManifestError("Synthetic reference-row count mismatch.")
    if len(real) != EXPECTED_REAL_REFERENCE_ROWS:
        raise AblationManifestError("Real reference-row count mismatch.")

    for row in values:
        if row.variant != REFERENCE_VARIANT:
            raise AblationManifestError("Reference row variant must be PC_NFPSO.")
        if len(row.external_run_id) != 64 or len(row.result_sha256) != 64:
            raise AblationManifestError("Invalid reference SHA-256 identity.")
        if row.domain == "synthetic":
            if row.dgp_seed not in SYNTHETIC_DGP_SEEDS:
                raise AblationManifestError("Synthetic reference DGP seed mismatch.")
            if row.noise_level not in CONFIRMATORY_NOISE_LEVELS:
                raise AblationManifestError("Synthetic reference noise mismatch.")
            if row.optimizer_seed not in ABLATION_OPTIMIZER_SEEDS:
                raise AblationManifestError("Synthetic reference optimizer seed mismatch.")
        elif row.domain == "real":
            if row.optimizer_seed not in REAL_REFERENCE_OPTIMIZER_SEEDS:
                raise AblationManifestError("Real reference optimizer seed mismatch.")
        else:
            raise AblationManifestError("Unexpected reference-row domain.")


def protocol_fingerprint(
    project_root: str | os.PathLike[str] = ".",
) -> str:
    root = Path(project_root)
    h = hashlib.sha256()
    for relative in PROTOCOL_INPUT_FILES:
        path = root / relative
        if not path.is_file():
            raise AblationManifestError(
                f"Missing ablation protocol input: {relative}"
            )
        h.update(relative.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def git_head(project_root: str | os.PathLike[str] = ".") -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(project_root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def assert_clean_git_worktree(
    project_root: str | os.PathLike[str] = ".",
) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=Path(project_root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise AblationManifestError("Ablation manifest preparation requires a clean Git tree.")


def build_ablation_manifest(
    *,
    project_root: str | os.PathLike[str],
    external_workspace: str | os.PathLike[str],
    protocol_fingerprint_value: str,
    code_commit: str,
) -> AblationManifest:
    tasks = build_new_ablation_tasks(project_root=project_root)
    references = build_reference_reuse_rows(
        external_workspace=external_workspace,
        project_root=project_root,
    )
    aggregate = external_results_aggregate_sha256(external_workspace)

    manifest = AblationManifest(
        schema_version=SCHEMA_VERSION,
        protocol_fingerprint=str(protocol_fingerprint_value),
        code_commit=str(code_commit),
        selected_radius=EXPECTED_RADIUS,
        selected_alpha=EXPECTED_ALPHA,
        synthetic_dgp_seeds=SYNTHETIC_DGP_SEEDS,
        noise_levels=tuple(float(v) for v in CONFIRMATORY_NOISE_LEVELS),
        ablation_optimizer_seeds=ABLATION_OPTIMIZER_SEEDS,
        real_reference_optimizer_seeds=REAL_REFERENCE_OPTIMIZER_SEEDS,
        variants=tuple(VARIANT_ORDER),
        new_task_count=len(tasks),
        synthetic_new_task_count=sum(t.domain == "synthetic" for t in tasks),
        real_new_task_count=sum(t.domain == "real" for t in tasks),
        reference_row_count=len(references),
        synthetic_reference_row_count=sum(r.domain == "synthetic" for r in references),
        real_reference_row_count=sum(r.domain == "real" for r in references),
        external_final_results_aggregate_sha256=aggregate,
        tasks=tasks,
        reference_rows=references,
    )
    validate_ablation_manifest(manifest)
    return manifest


def validate_ablation_manifest(manifest: AblationManifest) -> None:
    if manifest.schema_version != SCHEMA_VERSION:
        raise AblationManifestError("Ablation manifest schema mismatch.")
    if len(manifest.code_commit) != 40:
        raise AblationManifestError("Ablation manifest code commit must be a full SHA-1.")
    if len(manifest.protocol_fingerprint) != 64:
        raise AblationManifestError("Ablation protocol fingerprint must be SHA-256.")
    if manifest.selected_radius != EXPECTED_RADIUS:
        raise AblationManifestError("Ablation radius differs from frozen Development selection.")
    if manifest.selected_alpha != EXPECTED_ALPHA:
        raise AblationManifestError("Ablation alpha differs from frozen Development selection.")
    if manifest.synthetic_dgp_seeds != SYNTHETIC_DGP_SEEDS:
        raise AblationManifestError("Ablation synthetic DGP seeds differ from A010.")
    if manifest.ablation_optimizer_seeds != ABLATION_OPTIMIZER_SEEDS:
        raise AblationManifestError("Ablation optimizer seeds differ from A010.")
    if manifest.real_reference_optimizer_seeds != REAL_REFERENCE_OPTIMIZER_SEEDS:
        raise AblationManifestError("Real PC reference seed set differs from A010.")
    if manifest.variants != tuple(VARIANT_ORDER):
        raise AblationManifestError("Ablation variant order differs from frozen contract.")
    if manifest.new_task_count != EXPECTED_NEW_TASKS:
        raise AblationManifestError("Ablation new-task count mismatch.")
    if manifest.synthetic_new_task_count != EXPECTED_SYNTHETIC_NEW_TASKS:
        raise AblationManifestError("Synthetic new-task count mismatch.")
    if manifest.real_new_task_count != EXPECTED_REAL_NEW_TASKS:
        raise AblationManifestError("Real new-task count mismatch.")
    if manifest.reference_row_count != EXPECTED_REFERENCE_ROWS:
        raise AblationManifestError("Reference-row count mismatch.")
    if manifest.synthetic_reference_row_count != EXPECTED_SYNTHETIC_REFERENCE_ROWS:
        raise AblationManifestError("Synthetic reference-row count mismatch.")
    if manifest.real_reference_row_count != EXPECTED_REAL_REFERENCE_ROWS:
        raise AblationManifestError("Real reference-row count mismatch.")
    if len(manifest.external_final_results_aggregate_sha256) != 64:
        raise AblationManifestError("External Final aggregate SHA-256 is malformed.")

    validate_new_ablation_tasks(manifest.tasks)
    validate_reference_reuse_rows(manifest.reference_rows)

    new_ids = {t.run_id for t in manifest.tasks}
    external_ids = {r.external_run_id for r in manifest.reference_rows}
    if new_ids & external_ids:
        raise AblationManifestError("New ablation task IDs overlap external Final reference IDs.")


def _manifest_dict(manifest: AblationManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "protocol_fingerprint": manifest.protocol_fingerprint,
        "code_commit": manifest.code_commit,
        "selected_radius": manifest.selected_radius,
        "selected_alpha": manifest.selected_alpha,
        "synthetic_dgp_seeds": list(manifest.synthetic_dgp_seeds),
        "noise_levels": list(manifest.noise_levels),
        "ablation_optimizer_seeds": list(manifest.ablation_optimizer_seeds),
        "real_reference_optimizer_seeds": list(manifest.real_reference_optimizer_seeds),
        "variants": list(manifest.variants),
        "new_task_count": manifest.new_task_count,
        "synthetic_new_task_count": manifest.synthetic_new_task_count,
        "real_new_task_count": manifest.real_new_task_count,
        "reference_row_count": manifest.reference_row_count,
        "synthetic_reference_row_count": manifest.synthetic_reference_row_count,
        "real_reference_row_count": manifest.real_reference_row_count,
        "external_final_results_aggregate_sha256": (
            manifest.external_final_results_aggregate_sha256
        ),
        "tasks": [asdict(v) for v in manifest.tasks],
        "reference_rows": [asdict(v) for v in manifest.reference_rows],
    }


def _manifest_from_dict(data: dict[str, Any]) -> AblationManifest:
    tasks = tuple(AblationRunTask(**v) for v in data["tasks"])
    refs = tuple(ReferenceReuseRow(**v) for v in data["reference_rows"])
    manifest = AblationManifest(
        schema_version=str(data["schema_version"]),
        protocol_fingerprint=str(data["protocol_fingerprint"]),
        code_commit=str(data["code_commit"]),
        selected_radius=float(data["selected_radius"]),
        selected_alpha=float(data["selected_alpha"]),
        synthetic_dgp_seeds=tuple(int(v) for v in data["synthetic_dgp_seeds"]),
        noise_levels=tuple(float(v) for v in data["noise_levels"]),
        ablation_optimizer_seeds=tuple(int(v) for v in data["ablation_optimizer_seeds"]),
        real_reference_optimizer_seeds=tuple(
            int(v) for v in data["real_reference_optimizer_seeds"]
        ),
        variants=tuple(str(v) for v in data["variants"]),
        new_task_count=int(data["new_task_count"]),
        synthetic_new_task_count=int(data["synthetic_new_task_count"]),
        real_new_task_count=int(data["real_new_task_count"]),
        reference_row_count=int(data["reference_row_count"]),
        synthetic_reference_row_count=int(data["synthetic_reference_row_count"]),
        real_reference_row_count=int(data["real_reference_row_count"]),
        external_final_results_aggregate_sha256=str(
            data["external_final_results_aggregate_sha256"]
        ),
        tasks=tasks,
        reference_rows=refs,
    )
    validate_ablation_manifest(manifest)
    return manifest


def manifest_bytes(manifest: AblationManifest) -> bytes:
    validate_ablation_manifest(manifest)
    return (
        json.dumps(
            _manifest_dict(manifest),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_ablation_manifest(
    output_dir: str | os.PathLike[str],
    manifest: AblationManifest,
) -> tuple[Path, Path]:
    out = Path(output_dir)
    manifest_path = out / MANIFEST_FILENAME
    sha_path = out / MANIFEST_SHA_FILENAME
    if manifest_path.exists() or sha_path.exists():
        raise AblationManifestError("Refusing to overwrite an ablation manifest.")
    payload = manifest_bytes(manifest)
    digest = hashlib.sha256(payload).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(payload)
    sha_payload = (
        json.dumps(
            {MANIFEST_FILENAME: digest},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    sha_path.write_bytes(sha_payload)
    return manifest_path, sha_path


def load_ablation_manifest(
    output_dir: str | os.PathLike[str],
    *,
    expected_protocol_fingerprint: str | None = None,
    expected_code_commit: str | None = None,
) -> AblationManifest:
    out = Path(output_dir)
    manifest_path = out / MANIFEST_FILENAME
    sha_path = out / MANIFEST_SHA_FILENAME
    if not manifest_path.is_file() or not sha_path.is_file():
        raise AblationManifestError("Ablation manifest or SHA sidecar is missing.")

    sidecar = json.loads(sha_path.read_text(encoding="utf-8"))
    expected_sha = str(sidecar.get(MANIFEST_FILENAME, ""))
    observed_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if observed_sha != expected_sha:
        raise AblationManifestError("Ablation manifest SHA-256 mismatch.")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _manifest_from_dict(data)
    if (
        expected_protocol_fingerprint is not None
        and manifest.protocol_fingerprint != expected_protocol_fingerprint
    ):
        raise AblationManifestError("Ablation protocol fingerprint mismatch.")
    if (
        expected_code_commit is not None
        and manifest.code_commit != expected_code_commit
    ):
        raise AblationManifestError("Ablation code commit mismatch.")
    return manifest


def prepare_ablation_manifest(
    output_dir: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str],
    external_workspace: str | os.PathLike[str],
) -> AblationManifest:
    assert_clean_git_worktree(project_root)
    fingerprint = protocol_fingerprint(project_root)
    head = git_head(project_root)
    out = Path(output_dir)

    if (out / MANIFEST_FILENAME).exists():
        return load_ablation_manifest(
            out,
            expected_protocol_fingerprint=fingerprint,
            expected_code_commit=head,
        )

    manifest = build_ablation_manifest(
        project_root=project_root,
        external_workspace=external_workspace,
        protocol_fingerprint_value=fingerprint,
        code_commit=head,
    )
    write_ablation_manifest(out, manifest)
    return manifest


__all__ = [
    "SCHEMA_VERSION",
    "TASK_NAMESPACE",
    "EXPECTED_RADIUS",
    "EXPECTED_ALPHA",
    "SYNTHETIC_DGP_SEEDS",
    "ABLATION_OPTIMIZER_SEEDS",
    "REAL_REFERENCE_OPTIMIZER_SEEDS",
    "NEW_STOCHASTIC_VARIANTS",
    "REFERENCE_VARIANT",
    "EXPECTED_NEW_TASKS",
    "EXPECTED_REFERENCE_ROWS",
    "AblationManifestError",
    "AblationRunTask",
    "ReferenceReuseRow",
    "AblationManifest",
    "deterministic_ablation_run_id",
    "build_new_ablation_tasks",
    "validate_new_ablation_tasks",
    "select_reference_final_tasks",
    "build_reference_reuse_rows",
    "validate_reference_reuse_rows",
    "external_results_aggregate_sha256",
    "protocol_fingerprint",
    "build_ablation_manifest",
    "validate_ablation_manifest",
    "manifest_bytes",
    "write_ablation_manifest",
    "load_ablation_manifest",
    "prepare_ablation_manifest",
]
