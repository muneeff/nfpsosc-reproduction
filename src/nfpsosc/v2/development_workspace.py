"""
Execution-environment and workspace freeze for the V2 synthetic development selection.

This module does not execute PC-NFPSO development runs.  It freezes and verifies
the execution workspace that must exist before the first development outcome is
generated.

The scientific design remains defined by the Gate-2 protocol, amendments A004-A007,
and ``development_runner.py``.  This layer records execution provenance, validates
the pinned software environment, binds the workspace to the committed code and
protocol fingerprint, verifies the complete 28,350-task manifest, and fail-closes
on tampering or unexpected pre-existing result files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Mapping

from .development_runner import (
    SCHEMA_VERSION as DEVELOPMENT_RUNNER_SCHEMA_VERSION,
    EXPECTED_CONDITIONS,
    EXPECTED_RUNS,
    DevelopmentManifest,
    DevelopmentRunnerError,
    assert_clean_git_worktree,
    collect_development_results,
    git_head,
    load_development_manifest,
    pending_development_tasks,
    prepare_development_workspace,
    protocol_fingerprint,
)


WORKSPACE_SCHEMA_VERSION = "v2-development-workspace-freeze-1"
WORKSPACE_FREEZE_FILENAME = "workspace_freeze.json"
WORKSPACE_FREEZE_HASH_FILENAME = "workspace_freeze.sha256"
DEVELOPMENT_MANIFEST_FILENAME = "development_manifest.json"
DEVELOPMENT_MANIFEST_HASH_FILENAME = "development_manifest.sha256"
RESULTS_DIRECTORY_NAME = "results"

EXPECTED_PYTHON_VERSION = "3.10.11"
EXPECTED_PACKAGE_VERSIONS: Mapping[str, str] = {
    "numpy": "2.2.6",
    "pandas": "2.3.3",
    "scipy": "1.15.3",
    "statsmodels": "0.14.6",
    "scikit-learn": "1.7.2",
    "xgboost": "3.2.0",
}


class DevelopmentWorkspaceError(RuntimeError):
    """Raised when the frozen development workspace contract is violated."""


@dataclass(frozen=True)
class EnvironmentSnapshot:
    python_version: str
    python_implementation: str
    executable: str
    packages: dict[str, str]
    platform: str


@dataclass(frozen=True)
class DevelopmentWorkspaceFreeze:
    schema_version: str
    frozen_at_utc: str
    code_commit: str
    protocol_fingerprint: str
    development_runner_schema_version: str
    development_manifest_sha256: str
    task_count: int
    condition_count: int
    generators: tuple[str, ...]
    dgp_seeds: tuple[int, ...]
    noise_levels: tuple[float, ...]
    optimizer_seeds: tuple[int, ...]
    candidate_pairs: tuple[tuple[float, float], ...]
    first_run_id: str
    last_run_id: str
    result_file_pattern: str
    selection_success_count_required: int
    environment: EnvironmentSnapshot


@dataclass(frozen=True)
class DevelopmentWorkspaceVerification:
    code_commit: str
    protocol_fingerprint: str
    manifest_sha256: str
    task_count: int
    completed_count: int
    successful_count: int
    failed_count: int
    pending_count: int
    first_run_id: str
    last_run_id: str
    ready_for_first_run: bool


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    try:
        with open(tmp, "xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_environment() -> EnvironmentSnapshot:
    packages: dict[str, str] = {}
    for distribution in EXPECTED_PACKAGE_VERSIONS:
        try:
            packages[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError as exc:
            raise DevelopmentWorkspaceError(
                f"Required frozen package is not installed: {distribution}."
            ) from exc
    return EnvironmentSnapshot(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        executable=str(Path(sys.executable).resolve()),
        packages=packages,
        platform=platform.platform(),
    )


def assert_frozen_environment(snapshot: EnvironmentSnapshot) -> None:
    if snapshot.python_version != EXPECTED_PYTHON_VERSION:
        raise DevelopmentWorkspaceError(
            "Python version mismatch: "
            f"expected {EXPECTED_PYTHON_VERSION}, got {snapshot.python_version}."
        )
    if snapshot.python_implementation != "CPython":
        raise DevelopmentWorkspaceError(
            f"Python implementation must be CPython, got {snapshot.python_implementation!r}."
        )
    for distribution, expected in EXPECTED_PACKAGE_VERSIONS.items():
        observed = snapshot.packages.get(distribution)
        if observed != expected:
            raise DevelopmentWorkspaceError(
                f"Package version mismatch for {distribution}: "
                f"expected {expected}, got {observed!r}."
            )


def _environment_to_dict(snapshot: EnvironmentSnapshot) -> dict[str, object]:
    return {
        "python_version": snapshot.python_version,
        "python_implementation": snapshot.python_implementation,
        "executable": snapshot.executable,
        "packages": dict(snapshot.packages),
        "platform": snapshot.platform,
    }


def _environment_from_dict(data: Mapping[str, object]) -> EnvironmentSnapshot:
    packages_raw = data.get("packages")
    if not isinstance(packages_raw, dict):
        raise DevelopmentWorkspaceError("Workspace environment package map is invalid.")
    return EnvironmentSnapshot(
        python_version=str(data.get("python_version", "")),
        python_implementation=str(data.get("python_implementation", "")),
        executable=str(data.get("executable", "")),
        packages={str(k): str(v) for k, v in packages_raw.items()},
        platform=str(data.get("platform", "")),
    )


def _freeze_to_dict(freeze: DevelopmentWorkspaceFreeze) -> dict[str, object]:
    return {
        "schema_version": freeze.schema_version,
        "frozen_at_utc": freeze.frozen_at_utc,
        "code_commit": freeze.code_commit,
        "protocol_fingerprint": freeze.protocol_fingerprint,
        "development_runner_schema_version": freeze.development_runner_schema_version,
        "development_manifest_sha256": freeze.development_manifest_sha256,
        "task_count": freeze.task_count,
        "condition_count": freeze.condition_count,
        "generators": list(freeze.generators),
        "dgp_seeds": list(freeze.dgp_seeds),
        "noise_levels": list(freeze.noise_levels),
        "optimizer_seeds": list(freeze.optimizer_seeds),
        "candidate_pairs": [
            {"radius": float(radius), "alpha": float(alpha)}
            for radius, alpha in freeze.candidate_pairs
        ],
        "first_run_id": freeze.first_run_id,
        "last_run_id": freeze.last_run_id,
        "result_file_pattern": freeze.result_file_pattern,
        "selection_success_count_required": freeze.selection_success_count_required,
        "environment": _environment_to_dict(freeze.environment),
    }


def _freeze_from_dict(data: Mapping[str, object]) -> DevelopmentWorkspaceFreeze:
    try:
        pair_rows = data["candidate_pairs"]
        pairs = tuple(
            (float(row["radius"]), float(row["alpha"]))  # type: ignore[index]
            for row in pair_rows  # type: ignore[union-attr]
        )
        freeze = DevelopmentWorkspaceFreeze(
            schema_version=str(data["schema_version"]),
            frozen_at_utc=str(data["frozen_at_utc"]),
            code_commit=str(data["code_commit"]),
            protocol_fingerprint=str(data["protocol_fingerprint"]),
            development_runner_schema_version=str(
                data["development_runner_schema_version"]
            ),
            development_manifest_sha256=str(data["development_manifest_sha256"]),
            task_count=int(data["task_count"]),
            condition_count=int(data["condition_count"]),
            generators=tuple(str(v) for v in data["generators"]),  # type: ignore[union-attr]
            dgp_seeds=tuple(int(v) for v in data["dgp_seeds"]),  # type: ignore[union-attr]
            noise_levels=tuple(float(v) for v in data["noise_levels"]),  # type: ignore[union-attr]
            optimizer_seeds=tuple(int(v) for v in data["optimizer_seeds"]),  # type: ignore[union-attr]
            candidate_pairs=pairs,
            first_run_id=str(data["first_run_id"]),
            last_run_id=str(data["last_run_id"]),
            result_file_pattern=str(data["result_file_pattern"]),
            selection_success_count_required=int(
                data["selection_success_count_required"]
            ),
            environment=_environment_from_dict(data["environment"]),  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DevelopmentWorkspaceError("Workspace freeze payload is malformed.") from exc
    validate_workspace_freeze(freeze)
    return freeze


def validate_workspace_freeze(freeze: DevelopmentWorkspaceFreeze) -> None:
    if freeze.schema_version != WORKSPACE_SCHEMA_VERSION:
        raise DevelopmentWorkspaceError("Workspace freeze schema mismatch.")
    if len(freeze.code_commit) != 40:
        raise DevelopmentWorkspaceError("Workspace code commit must be a full git SHA-1.")
    if len(freeze.protocol_fingerprint) != 64:
        raise DevelopmentWorkspaceError("Workspace protocol fingerprint must be SHA-256.")
    if len(freeze.development_manifest_sha256) != 64:
        raise DevelopmentWorkspaceError("Workspace manifest digest must be SHA-256.")
    if freeze.development_runner_schema_version != DEVELOPMENT_RUNNER_SCHEMA_VERSION:
        raise DevelopmentWorkspaceError("Development runner schema mismatch.")
    if freeze.task_count != EXPECTED_RUNS:
        raise DevelopmentWorkspaceError("Workspace task count mismatch.")
    if freeze.condition_count != EXPECTED_CONDITIONS:
        raise DevelopmentWorkspaceError("Workspace condition count mismatch.")
    if freeze.selection_success_count_required != EXPECTED_RUNS:
        raise DevelopmentWorkspaceError("Workspace selection success gate mismatch.")
    if not freeze.generators or not freeze.dgp_seeds or not freeze.noise_levels:
        raise DevelopmentWorkspaceError("Workspace design coverage cannot be empty.")
    if not freeze.optimizer_seeds or not freeze.candidate_pairs:
        raise DevelopmentWorkspaceError("Workspace optimizer/grid coverage cannot be empty.")
    if len(freeze.first_run_id) != 64 or len(freeze.last_run_id) != 64:
        raise DevelopmentWorkspaceError("Workspace first/last run IDs must be SHA-256.")
    if freeze.result_file_pattern != "results/<run_id>.json":
        raise DevelopmentWorkspaceError("Unexpected result checkpoint layout.")
    assert_frozen_environment(freeze.environment)


def _manifest_digest_from_sidecar(output_dir: Path) -> str:
    manifest_path = output_dir / DEVELOPMENT_MANIFEST_FILENAME
    sidecar = output_dir / DEVELOPMENT_MANIFEST_HASH_FILENAME
    if not manifest_path.is_file() or not sidecar.is_file():
        raise DevelopmentWorkspaceError("Development manifest or hash sidecar is missing.")
    expected = sidecar.read_text(encoding="ascii").strip()
    if len(expected) != 64:
        raise DevelopmentWorkspaceError("Development manifest hash sidecar is invalid.")
    observed = sha256_file(manifest_path)
    if observed != expected:
        raise DevelopmentWorkspaceError("Development manifest SHA-256 mismatch.")
    return observed


def _candidate_pairs(manifest: DevelopmentManifest) -> tuple[tuple[float, float], ...]:
    return tuple(
        (float(pair.radius), float(pair.alpha))
        for pair in manifest.candidate_pairs
    )


def _assert_manifest_matches_freeze(
    manifest: DevelopmentManifest,
    freeze: DevelopmentWorkspaceFreeze,
) -> None:
    if manifest.code_commit != freeze.code_commit:
        raise DevelopmentWorkspaceError("Manifest code commit differs from workspace freeze.")
    if manifest.protocol_fingerprint != freeze.protocol_fingerprint:
        raise DevelopmentWorkspaceError(
            "Manifest protocol fingerprint differs from workspace freeze."
        )
    if manifest.task_count != freeze.task_count:
        raise DevelopmentWorkspaceError("Manifest task count differs from workspace freeze.")
    if manifest.condition_count != freeze.condition_count:
        raise DevelopmentWorkspaceError(
            "Manifest condition count differs from workspace freeze."
        )
    if tuple(manifest.generators) != freeze.generators:
        raise DevelopmentWorkspaceError("Manifest generator coverage differs from workspace freeze.")
    if tuple(manifest.dgp_seeds) != freeze.dgp_seeds:
        raise DevelopmentWorkspaceError("Manifest DGP seeds differ from workspace freeze.")
    if tuple(manifest.noise_levels) != freeze.noise_levels:
        raise DevelopmentWorkspaceError("Manifest noise levels differ from workspace freeze.")
    if tuple(manifest.optimizer_seeds) != freeze.optimizer_seeds:
        raise DevelopmentWorkspaceError(
            "Manifest optimizer seeds differ from workspace freeze."
        )
    if _candidate_pairs(manifest) != freeze.candidate_pairs:
        raise DevelopmentWorkspaceError("Manifest candidate grid differs from workspace freeze.")
    if not manifest.tasks:
        raise DevelopmentWorkspaceError("Development manifest contains no tasks.")
    if manifest.tasks[0].run_id != freeze.first_run_id:
        raise DevelopmentWorkspaceError("Manifest first run ID differs from workspace freeze.")
    if manifest.tasks[-1].run_id != freeze.last_run_id:
        raise DevelopmentWorkspaceError("Manifest last run ID differs from workspace freeze.")


def load_workspace_freeze(
    output_dir: str | os.PathLike[str],
) -> DevelopmentWorkspaceFreeze:
    out = Path(output_dir)
    freeze_path = out / WORKSPACE_FREEZE_FILENAME
    hash_path = out / WORKSPACE_FREEZE_HASH_FILENAME
    if not freeze_path.is_file() or not hash_path.is_file():
        raise DevelopmentWorkspaceError("Workspace freeze or hash sidecar is missing.")
    payload = freeze_path.read_bytes()
    expected = hash_path.read_text(encoding="ascii").strip()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise DevelopmentWorkspaceError("Workspace freeze SHA-256 mismatch.")
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DevelopmentWorkspaceError("Workspace freeze is not valid JSON.") from exc
    if not isinstance(data, dict):
        raise DevelopmentWorkspaceError("Workspace freeze root must be a JSON object.")
    return _freeze_from_dict(data)


def _existing_result_json_paths(output_dir: Path) -> tuple[Path, ...]:
    result_dir = output_dir / RESULTS_DIRECTORY_NAME
    if not result_dir.exists():
        return ()
    if not result_dir.is_dir():
        raise DevelopmentWorkspaceError("Development results path is not a directory.")
    return tuple(sorted(result_dir.glob("*.json")))


def _assert_no_orphan_result_files(
    output_dir: Path,
    manifest: DevelopmentManifest,
) -> None:
    expected = {task.run_id for task in manifest.tasks}
    for path in _existing_result_json_paths(output_dir):
        if path.stem not in expected:
            raise DevelopmentWorkspaceError(
                f"Orphan development result file is not present in manifest: {path.name}"
            )


def freeze_development_workspace(
    output_dir: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str],
) -> DevelopmentWorkspaceFreeze:
    """
    Create the pre-outcome workspace freeze.

    This function refuses to create a new freeze if any development result JSON
    already exists.  Re-running it after a freeze exists verifies and returns the
    existing freeze instead of overwriting provenance.
    """
    out = Path(output_dir)
    freeze_path = out / WORKSPACE_FREEZE_FILENAME
    freeze_hash_path = out / WORKSPACE_FREEZE_HASH_FILENAME

    if freeze_path.exists() or freeze_hash_path.exists():
        if not (freeze_path.is_file() and freeze_hash_path.is_file()):
            raise DevelopmentWorkspaceError(
                "Partial workspace freeze exists; refusing implicit repair."
            )
        verification = verify_development_workspace(
            out,
            project_root=project_root,
            require_no_results=False,
        )
        freeze = load_workspace_freeze(out)
        if verification.code_commit != freeze.code_commit:
            raise DevelopmentWorkspaceError("Existing workspace verification failed.")
        return freeze

    assert_clean_git_worktree(project_root)
    environment = capture_environment()
    assert_frozen_environment(environment)

    manifest = prepare_development_workspace(
        out,
        project_root=project_root,
    )
    current_head = git_head(project_root)
    current_protocol = protocol_fingerprint(project_root)
    if manifest.code_commit != current_head:
        raise DevelopmentWorkspaceError("Prepared manifest is not bound to current HEAD.")
    if manifest.protocol_fingerprint != current_protocol:
        raise DevelopmentWorkspaceError(
            "Prepared manifest is not bound to current protocol fingerprint."
        )
    if manifest.task_count != EXPECTED_RUNS or manifest.condition_count != EXPECTED_CONDITIONS:
        raise DevelopmentWorkspaceError("Prepared development manifest counts are invalid.")
    if not manifest.tasks:
        raise DevelopmentWorkspaceError("Prepared development manifest contains no tasks.")

    _assert_no_orphan_result_files(out, manifest)
    existing = collect_development_results(out, manifest)
    if existing:
        raise DevelopmentWorkspaceError(
            "Cannot freeze a pre-outcome workspace after development results already exist."
        )
    pending = pending_development_tasks(out, manifest)
    if len(pending) != EXPECTED_RUNS:
        raise DevelopmentWorkspaceError(
            "Pre-outcome workspace must have all development runs pending."
        )

    manifest_digest = _manifest_digest_from_sidecar(out)
    freeze = DevelopmentWorkspaceFreeze(
        schema_version=WORKSPACE_SCHEMA_VERSION,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        code_commit=current_head,
        protocol_fingerprint=current_protocol,
        development_runner_schema_version=DEVELOPMENT_RUNNER_SCHEMA_VERSION,
        development_manifest_sha256=manifest_digest,
        task_count=manifest.task_count,
        condition_count=manifest.condition_count,
        generators=tuple(manifest.generators),
        dgp_seeds=tuple(manifest.dgp_seeds),
        noise_levels=tuple(manifest.noise_levels),
        optimizer_seeds=tuple(manifest.optimizer_seeds),
        candidate_pairs=_candidate_pairs(manifest),
        first_run_id=manifest.tasks[0].run_id,
        last_run_id=manifest.tasks[-1].run_id,
        result_file_pattern="results/<run_id>.json",
        selection_success_count_required=EXPECTED_RUNS,
        environment=environment,
    )
    validate_workspace_freeze(freeze)

    payload = _canonical_json_bytes(_freeze_to_dict(freeze))
    digest = hashlib.sha256(payload).hexdigest()
    _atomic_write_bytes(freeze_path, payload)
    try:
        _atomic_write_bytes(freeze_hash_path, (digest + "\n").encode("ascii"))
    except Exception:
        freeze_path.unlink(missing_ok=True)
        raise

    return load_workspace_freeze(out)


def verify_development_workspace(
    output_dir: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str],
    require_no_results: bool = False,
) -> DevelopmentWorkspaceVerification:
    out = Path(output_dir)
    assert_clean_git_worktree(project_root)

    freeze = load_workspace_freeze(out)
    current_head = git_head(project_root)
    current_protocol = protocol_fingerprint(project_root)
    if current_head != freeze.code_commit:
        raise DevelopmentWorkspaceError(
            f"Workspace is bound to code commit {freeze.code_commit}, "
            f"but current HEAD is {current_head}."
        )
    if current_protocol != freeze.protocol_fingerprint:
        raise DevelopmentWorkspaceError("Current protocol fingerprint differs from workspace freeze.")

    current_environment = capture_environment()
    assert_frozen_environment(current_environment)
    if current_environment.python_version != freeze.environment.python_version:
        raise DevelopmentWorkspaceError("Current Python version differs from workspace freeze.")
    if current_environment.python_implementation != freeze.environment.python_implementation:
        raise DevelopmentWorkspaceError(
            "Current Python implementation differs from workspace freeze."
        )
    if current_environment.packages != freeze.environment.packages:
        raise DevelopmentWorkspaceError(
            "Current frozen package versions differ from workspace freeze."
        )

    manifest_digest = _manifest_digest_from_sidecar(out)
    if manifest_digest != freeze.development_manifest_sha256:
        raise DevelopmentWorkspaceError(
            "Development manifest digest differs from workspace freeze."
        )
    try:
        manifest = load_development_manifest(
            out,
            expected_protocol_fingerprint=freeze.protocol_fingerprint,
            expected_code_commit=freeze.code_commit,
        )
    except DevelopmentRunnerError as exc:
        raise DevelopmentWorkspaceError(
            "Development manifest failed runner-level verification."
        ) from exc

    _assert_manifest_matches_freeze(manifest, freeze)
    _assert_no_orphan_result_files(out, manifest)

    results = collect_development_results(out, manifest)
    completed_count = len(results)
    failed_count = sum(1 for result in results if not result.successful)
    successful_count = completed_count - failed_count
    pending = pending_development_tasks(out, manifest)
    pending_count = len(pending)

    if completed_count + pending_count != EXPECTED_RUNS:
        raise DevelopmentWorkspaceError(
            "Completed plus pending development runs does not equal frozen task count."
        )
    if require_no_results and completed_count != 0:
        raise DevelopmentWorkspaceError(
            "First-run gate requires zero pre-existing development results."
        )

    return DevelopmentWorkspaceVerification(
        code_commit=freeze.code_commit,
        protocol_fingerprint=freeze.protocol_fingerprint,
        manifest_sha256=freeze.development_manifest_sha256,
        task_count=freeze.task_count,
        completed_count=completed_count,
        successful_count=successful_count,
        failed_count=failed_count,
        pending_count=pending_count,
        first_run_id=freeze.first_run_id,
        last_run_id=freeze.last_run_id,
        ready_for_first_run=(completed_count == 0 and pending_count == EXPECTED_RUNS),
    )


def assert_ready_for_first_development_run(
    output_dir: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str],
) -> DevelopmentWorkspaceVerification:
    verification = verify_development_workspace(
        output_dir,
        project_root=project_root,
        require_no_results=True,
    )
    if not verification.ready_for_first_run:
        raise DevelopmentWorkspaceError(
            "Development workspace is not in the frozen zero-outcome first-run state."
        )
    return verification


__all__ = [
    "WORKSPACE_SCHEMA_VERSION",
    "WORKSPACE_FREEZE_FILENAME",
    "WORKSPACE_FREEZE_HASH_FILENAME",
    "EXPECTED_PYTHON_VERSION",
    "EXPECTED_PACKAGE_VERSIONS",
    "DevelopmentWorkspaceError",
    "EnvironmentSnapshot",
    "DevelopmentWorkspaceFreeze",
    "DevelopmentWorkspaceVerification",
    "sha256_file",
    "capture_environment",
    "assert_frozen_environment",
    "validate_workspace_freeze",
    "load_workspace_freeze",
    "freeze_development_workspace",
    "verify_development_workspace",
    "assert_ready_for_first_development_run",
]
