from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Callable, Iterable, Literal, Sequence

import numpy as np

from .development_selection import (
    CandidatePair,
    CONFIRMATORY_NOISE_LEVELS,
    DEVELOPMENT_DGP_SEEDS,
    DEVELOPMENT_OPTIMIZER_SEEDS,
    DevelopmentScore,
    DevelopmentSelectionError,
    assert_development_seed_firewall,
    frozen_candidate_grid,
    select_global_radius_alpha,
)
from .pc_nfpso import PCNFPSOFit, fit_pc_nfpso_v2, forecast_pc_nfpso_v2
from .synthetic import ALL_GENERATORS, SERIES_LENGTH, generate_synthetic_series, seasonal_period


SCHEMA_VERSION = "v2-development-runner-1"
PRETEST_LENGTH = 144
TEST_LENGTH = 36
EXPECTED_GENERATORS = 9
EXPECTED_DGP_SEEDS = 10
EXPECTED_NOISE_LEVELS = 3
EXPECTED_OPTIMIZER_SEEDS = 3
EXPECTED_CANDIDATE_PAIRS = 35
EXPECTED_CONDITIONS = 270
EXPECTED_RUNS = 28_350
MIN_MASE_SCALE = 1e-12

PROTOCOL_INPUT_FILES: tuple[str, ...] = (
    "configs/v2/protocol_v2.json",
    "configs/v2/synthetic_split_v2.json",
    "configs/v2/synthetic_dgps_v2.json",
    "configs/v2/canonical_synthetic_v2.json",
    "configs/v2/amendment_004_tsk_implementation.json",
    "configs/v2/amendment_005_canonical_synthetic.json",
    "configs/v2/amendment_006_lorenz_rk4_order.json",
    "configs/v2/amendment_007_pcnfpso_validation_failure.json",
)


class DevelopmentRunnerError(RuntimeError):
    """Raised when the development runner violates the frozen V2 execution contract."""


@dataclass(frozen=True, order=True)
class DevelopmentRunTask:
    run_id: str
    generator: str
    dgp_seed: int
    noise_level: float
    optimizer_seed: int
    radius: float
    alpha: float

    @property
    def pair(self) -> CandidatePair:
        return CandidatePair(float(self.radius), float(self.alpha))

    @property
    def condition(self) -> tuple[str, int, float]:
        return (self.generator, int(self.dgp_seed), float(self.noise_level))


@dataclass(frozen=True)
class DevelopmentRunFailure:
    stage: Literal["generation", "fit", "forecast", "metrics"]
    error_type: str
    message: str


@dataclass(frozen=True)
class DevelopmentRunResult:
    task: DevelopmentRunTask
    status: Literal["success", "failed"]
    mase: float | None
    mase_scale: float | None
    n_lags: int
    validation_size: int
    seasonal_period: int
    best_objective: float | None
    n_rules: int | None
    failure: DevelopmentRunFailure | None

    @property
    def successful(self) -> bool:
        return self.status == "success"

    def to_score(self) -> DevelopmentScore:
        if not self.successful or self.mase is None:
            raise DevelopmentRunnerError(
                f"Failed development run {self.task.run_id} cannot supply a selection score."
            )
        return DevelopmentScore(
            generator=self.task.generator,
            dgp_seed=self.task.dgp_seed,
            noise_level=self.task.noise_level,
            optimizer_seed=self.task.optimizer_seed,
            radius=self.task.radius,
            alpha=self.task.alpha,
            mase=float(self.mase),
        )


@dataclass(frozen=True)
class DevelopmentManifest:
    schema_version: str
    protocol_fingerprint: str
    code_commit: str
    task_count: int
    condition_count: int
    generators: tuple[str, ...]
    dgp_seeds: tuple[int, ...]
    noise_levels: tuple[float, ...]
    optimizer_seeds: tuple[int, ...]
    candidate_pairs: tuple[CandidatePair, ...]
    tasks: tuple[DevelopmentRunTask, ...]


@dataclass(frozen=True)
class DevelopmentDryRunSummary:
    task_count: int
    condition_count: int
    generator_count: int
    candidate_pair_count: int
    dgp_seed_count: int
    noise_level_count: int
    optimizer_seed_count: int
    first_run_id: str
    last_run_id: str


def frozen_lag_dimension(seasonal_period_value: int) -> int:
    if isinstance(seasonal_period_value, (bool, np.bool_)) or not isinstance(
        seasonal_period_value, (int, np.integer)
    ):
        raise DevelopmentRunnerError("seasonal period must be an integer.")
    m = int(seasonal_period_value)
    if m < 1:
        raise DevelopmentRunnerError("seasonal period must be at least one.")
    return min(12, max(5, m))


def pc_nfpso_validation_size(pretest_length: int, n_lags: int) -> int:
    if pretest_length < 1 or n_lags < 1 or pretest_length <= n_lags:
        raise DevelopmentRunnerError("Invalid pre-test length or lag dimension.")
    n_supervised = int(pretest_length) - int(n_lags)
    validation = max(8, int(np.floor(0.20 * n_supervised)))
    if n_supervised - validation < 11:
        raise DevelopmentRunnerError(
            "A007 requires at least 11 fitting supervised rows after validation."
        )
    return int(validation)


def raw_mase_scale(raw_pretest: np.ndarray, *, seasonal_period_value: int) -> float:
    y = np.asarray(raw_pretest, dtype=float).reshape(-1)
    if len(y) < 2 or not np.all(np.isfinite(y)):
        raise DevelopmentRunnerError("MASE pre-test series must be finite and non-empty.")
    m = int(seasonal_period_value)
    if m < 1 or len(y) <= m:
        raise DevelopmentRunnerError("Invalid seasonal period for MASE scaling.")
    scale = float(np.mean(np.abs(y[m:] - y[:-m])))
    if not np.isfinite(scale) or scale <= MIN_MASE_SCALE:
        raise DevelopmentRunnerError(
            f"MASE denominator must exceed {MIN_MASE_SCALE}; received {scale!r}."
        )
    return scale


def mase_from_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    mase_scale: float,
) -> float:
    y = np.asarray(actual, dtype=float).reshape(-1)
    p = np.asarray(predicted, dtype=float).reshape(-1)
    if y.shape != p.shape or len(y) == 0:
        raise DevelopmentRunnerError("Actual/predicted arrays must have equal non-zero length.")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
        raise DevelopmentRunnerError("Actual/predicted arrays must be finite.")
    scale = float(mase_scale)
    if not np.isfinite(scale) or scale <= MIN_MASE_SCALE:
        raise DevelopmentRunnerError("Invalid MASE scale.")
    mase = float(np.mean(np.abs(y - p)) / scale)
    if not np.isfinite(mase) or mase < 0.0:
        raise DevelopmentRunnerError("Computed MASE is invalid.")
    return mase


def _canonical_float(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        raise DevelopmentRunnerError("Run-key floating values must be finite.")
    return repr(value)


def deterministic_run_id(
    generator: str,
    dgp_seed: int,
    noise_level: float,
    optimizer_seed: int,
    radius: float,
    alpha: float,
) -> str:
    payload = {
        "namespace": "pc-nfpso-v2-development",
        "generator": str(generator),
        "dgp_seed": int(dgp_seed),
        "noise_level": _canonical_float(noise_level),
        "optimizer_seed": int(optimizer_seed),
        "radius": _canonical_float(radius),
        "alpha": _canonical_float(alpha),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=1)
def build_development_tasks() -> tuple[DevelopmentRunTask, ...]:
    tasks: list[DevelopmentRunTask] = []
    for generator in ALL_GENERATORS:
        for dgp_seed in DEVELOPMENT_DGP_SEEDS:
            for noise_level in CONFIRMATORY_NOISE_LEVELS:
                for pair in frozen_candidate_grid():
                    for optimizer_seed in DEVELOPMENT_OPTIMIZER_SEEDS:
                        run_id = deterministic_run_id(
                            generator,
                            dgp_seed,
                            noise_level,
                            optimizer_seed,
                            pair.radius,
                            pair.alpha,
                        )
                        tasks.append(
                            DevelopmentRunTask(
                                run_id=run_id,
                                generator=generator,
                                dgp_seed=int(dgp_seed),
                                noise_level=float(noise_level),
                                optimizer_seed=int(optimizer_seed),
                                radius=float(pair.radius),
                                alpha=float(pair.alpha),
                            )
                        )
    validate_development_tasks(tasks)
    return tuple(tasks)


def validate_development_tasks(tasks: Sequence[DevelopmentRunTask]) -> None:
    rows = tuple(tasks)
    if len(rows) != EXPECTED_RUNS:
        raise DevelopmentRunnerError(
            f"Development manifest must contain exactly {EXPECTED_RUNS} runs; got {len(rows)}."
        )

    run_ids = [row.run_id for row in rows]
    if len(set(run_ids)) != len(run_ids):
        raise DevelopmentRunnerError("Development manifest contains duplicate run IDs.")

    observed_keys: set[tuple[str, int, float, int, float, float]] = set()
    for row in rows:
        if row.generator not in ALL_GENERATORS:
            raise DevelopmentRunnerError(f"Unexpected generator {row.generator!r}.")
        assert_development_seed_firewall(
            row.dgp_seed, row.optimizer_seed, row.noise_level
        )
        if row.pair not in set(frozen_candidate_grid()):
            raise DevelopmentRunnerError(f"Pair outside frozen grid: {row.pair}.")
        expected_id = deterministic_run_id(
            row.generator,
            row.dgp_seed,
            row.noise_level,
            row.optimizer_seed,
            row.radius,
            row.alpha,
        )
        if row.run_id != expected_id:
            raise DevelopmentRunnerError(
                f"Run ID mismatch for development task {row.run_id}."
            )
        key = (
            row.generator,
            row.dgp_seed,
            row.noise_level,
            row.optimizer_seed,
            row.radius,
            row.alpha,
        )
        if key in observed_keys:
            raise DevelopmentRunnerError(f"Duplicate development task key: {key}.")
        observed_keys.add(key)

    expected_keys = {
        (
            generator,
            dgp_seed,
            noise_level,
            optimizer_seed,
            pair.radius,
            pair.alpha,
        )
        for generator in ALL_GENERATORS
        for dgp_seed in DEVELOPMENT_DGP_SEEDS
        for noise_level in CONFIRMATORY_NOISE_LEVELS
        for pair in frozen_candidate_grid()
        for optimizer_seed in DEVELOPMENT_OPTIMIZER_SEEDS
    }
    if observed_keys != expected_keys:
        raise DevelopmentRunnerError("Development manifest does not equal the frozen Cartesian design.")


def development_dry_run_summary() -> DevelopmentDryRunSummary:
    tasks = build_development_tasks()
    conditions = {task.condition for task in tasks}
    return DevelopmentDryRunSummary(
        task_count=len(tasks),
        condition_count=len(conditions),
        generator_count=len({task.generator for task in tasks}),
        candidate_pair_count=len({task.pair for task in tasks}),
        dgp_seed_count=len({task.dgp_seed for task in tasks}),
        noise_level_count=len({task.noise_level for task in tasks}),
        optimizer_seed_count=len({task.optimizer_seed for task in tasks}),
        first_run_id=tasks[0].run_id,
        last_run_id=tasks[-1].run_id,
    )


def protocol_fingerprint(project_root: str | os.PathLike[str]) -> str:
    root = Path(project_root)
    digest = hashlib.sha256()
    for relative in PROTOCOL_INPUT_FILES:
        path = root / relative
        if not path.is_file():
            raise DevelopmentRunnerError(f"Missing frozen protocol input: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_head(project_root: str | os.PathLike[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DevelopmentRunnerError("Unable to resolve git HEAD.") from exc
    head = completed.stdout.strip()
    if len(head) != 40:
        raise DevelopmentRunnerError(f"Unexpected git HEAD value: {head!r}")
    return head


def assert_clean_git_worktree(project_root: str | os.PathLike[str]) -> None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DevelopmentRunnerError("Unable to inspect git working tree.") from exc
    if completed.stdout.strip():
        raise DevelopmentRunnerError(
            "Development execution requires a clean git working tree."
        )


def build_development_manifest(
    *,
    protocol_fingerprint_value: str,
    code_commit: str,
) -> DevelopmentManifest:
    if not isinstance(protocol_fingerprint_value, str) or len(protocol_fingerprint_value) != 64:
        raise DevelopmentRunnerError("protocol_fingerprint must be a SHA-256 hex string.")
    if not isinstance(code_commit, str) or len(code_commit) != 40:
        raise DevelopmentRunnerError("code_commit must be a 40-character git commit hash.")
    tasks = build_development_tasks()
    return DevelopmentManifest(
        schema_version=SCHEMA_VERSION,
        protocol_fingerprint=protocol_fingerprint_value,
        code_commit=code_commit,
        task_count=len(tasks),
        condition_count=EXPECTED_CONDITIONS,
        generators=tuple(ALL_GENERATORS),
        dgp_seeds=tuple(DEVELOPMENT_DGP_SEEDS),
        noise_levels=tuple(CONFIRMATORY_NOISE_LEVELS),
        optimizer_seeds=tuple(DEVELOPMENT_OPTIMIZER_SEEDS),
        candidate_pairs=frozen_candidate_grid(),
        tasks=tasks,
    )


def _task_to_dict(task: DevelopmentRunTask) -> dict[str, object]:
    return asdict(task)


def _pair_to_dict(pair: CandidatePair) -> dict[str, float]:
    return {"radius": float(pair.radius), "alpha": float(pair.alpha)}


def _manifest_to_dict(manifest: DevelopmentManifest) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "protocol_fingerprint": manifest.protocol_fingerprint,
        "code_commit": manifest.code_commit,
        "task_count": manifest.task_count,
        "condition_count": manifest.condition_count,
        "generators": list(manifest.generators),
        "dgp_seeds": list(manifest.dgp_seeds),
        "noise_levels": list(manifest.noise_levels),
        "optimizer_seeds": list(manifest.optimizer_seeds),
        "candidate_pairs": [_pair_to_dict(pair) for pair in manifest.candidate_pairs],
        "tasks": [_task_to_dict(task) for task in manifest.tasks],
    }


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


def _canonical_json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def write_development_manifest(
    output_dir: str | os.PathLike[str],
    manifest: DevelopmentManifest,
) -> tuple[Path, Path]:
    validate_development_tasks(manifest.tasks)
    if manifest.task_count != EXPECTED_RUNS or manifest.condition_count != EXPECTED_CONDITIONS:
        raise DevelopmentRunnerError("Development manifest count metadata is invalid.")
    out = Path(output_dir)
    manifest_path = out / "development_manifest.json"
    hash_path = out / "development_manifest.sha256"
    payload = _canonical_json_bytes(_manifest_to_dict(manifest))
    digest = hashlib.sha256(payload).hexdigest()

    if manifest_path.exists() or hash_path.exists():
        raise DevelopmentRunnerError("Development manifest already exists; refusing overwrite.")

    _atomic_write_bytes(manifest_path, payload)
    try:
        _atomic_write_bytes(hash_path, (digest + "\n").encode("ascii"))
    except Exception:
        manifest_path.unlink(missing_ok=True)
        raise
    return manifest_path, hash_path


def _manifest_from_dict(data: dict[str, object]) -> DevelopmentManifest:
    tasks = tuple(DevelopmentRunTask(**row) for row in data["tasks"])  # type: ignore[arg-type]
    pairs = tuple(CandidatePair(**row) for row in data["candidate_pairs"])  # type: ignore[arg-type]
    manifest = DevelopmentManifest(
        schema_version=str(data["schema_version"]),
        protocol_fingerprint=str(data["protocol_fingerprint"]),
        code_commit=str(data["code_commit"]),
        task_count=int(data["task_count"]),
        condition_count=int(data["condition_count"]),
        generators=tuple(str(v) for v in data["generators"]),  # type: ignore[union-attr]
        dgp_seeds=tuple(int(v) for v in data["dgp_seeds"]),  # type: ignore[union-attr]
        noise_levels=tuple(float(v) for v in data["noise_levels"]),  # type: ignore[union-attr]
        optimizer_seeds=tuple(int(v) for v in data["optimizer_seeds"]),  # type: ignore[union-attr]
        candidate_pairs=pairs,
        tasks=tasks,
    )
    validate_development_tasks(manifest.tasks)
    if manifest.schema_version != SCHEMA_VERSION:
        raise DevelopmentRunnerError("Development manifest schema mismatch.")
    if manifest.task_count != EXPECTED_RUNS or manifest.condition_count != EXPECTED_CONDITIONS:
        raise DevelopmentRunnerError("Development manifest count metadata mismatch.")
    if manifest.generators != tuple(ALL_GENERATORS):
        raise DevelopmentRunnerError("Development manifest generator order/coverage mismatch.")
    if manifest.dgp_seeds != tuple(DEVELOPMENT_DGP_SEEDS):
        raise DevelopmentRunnerError("Development manifest DGP seed coverage mismatch.")
    if manifest.noise_levels != tuple(CONFIRMATORY_NOISE_LEVELS):
        raise DevelopmentRunnerError("Development manifest noise-level coverage mismatch.")
    if manifest.optimizer_seeds != tuple(DEVELOPMENT_OPTIMIZER_SEEDS):
        raise DevelopmentRunnerError("Development manifest optimizer seed coverage mismatch.")
    if manifest.candidate_pairs != frozen_candidate_grid():
        raise DevelopmentRunnerError("Development manifest candidate-grid mismatch.")
    return manifest


def load_development_manifest(
    output_dir: str | os.PathLike[str],
    *,
    expected_protocol_fingerprint: str | None = None,
    expected_code_commit: str | None = None,
) -> DevelopmentManifest:
    out = Path(output_dir)
    manifest_path = out / "development_manifest.json"
    hash_path = out / "development_manifest.sha256"
    if not manifest_path.is_file() or not hash_path.is_file():
        raise DevelopmentRunnerError("Development manifest or hash sidecar is missing.")
    payload = manifest_path.read_bytes()
    expected_hash = hash_path.read_text(encoding="ascii").strip()
    observed_hash = hashlib.sha256(payload).hexdigest()
    if observed_hash != expected_hash:
        raise DevelopmentRunnerError("Development manifest SHA-256 mismatch.")
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DevelopmentRunnerError("Development manifest is not valid JSON.") from exc
    manifest = _manifest_from_dict(data)
    if expected_protocol_fingerprint is not None and manifest.protocol_fingerprint != expected_protocol_fingerprint:
        raise DevelopmentRunnerError("Development manifest protocol fingerprint mismatch.")
    if expected_code_commit is not None and manifest.code_commit != expected_code_commit:
        raise DevelopmentRunnerError("Development manifest code commit mismatch.")
    return manifest


def prepare_development_workspace(
    output_dir: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str],
) -> DevelopmentManifest:
    assert_clean_git_worktree(project_root)
    fingerprint = protocol_fingerprint(project_root)
    head = git_head(project_root)
    out = Path(output_dir)
    manifest_path = out / "development_manifest.json"
    if manifest_path.exists():
        return load_development_manifest(
            out,
            expected_protocol_fingerprint=fingerprint,
            expected_code_commit=head,
        )
    manifest = build_development_manifest(
        protocol_fingerprint_value=fingerprint,
        code_commit=head,
    )
    write_development_manifest(out, manifest)
    return manifest


def _failure(task: DevelopmentRunTask, stage: str, exc: Exception, *, m: int, L: int, V: int) -> DevelopmentRunResult:
    return DevelopmentRunResult(
        task=task,
        status="failed",
        mase=None,
        mase_scale=None,
        n_lags=L,
        validation_size=V,
        seasonal_period=m,
        best_objective=None,
        n_rules=None,
        failure=DevelopmentRunFailure(
            stage=stage,  # type: ignore[arg-type]
            error_type=type(exc).__name__,
            message=str(exc),
        ),
    )


def execute_development_task(
    task: DevelopmentRunTask,
    *,
    fit_function: Callable[..., PCNFPSOFit] = fit_pc_nfpso_v2,
    forecast_function: Callable[[PCNFPSOFit, np.ndarray], np.ndarray] = forecast_pc_nfpso_v2,
) -> DevelopmentRunResult:
    assert_development_seed_firewall(task.dgp_seed, task.optimizer_seed, task.noise_level)
    if task.generator not in ALL_GENERATORS:
        raise DevelopmentRunnerError(f"Unexpected generator {task.generator!r}.")
    if task.pair not in set(frozen_candidate_grid()):
        raise DevelopmentRunnerError(f"Pair outside frozen grid: {task.pair}.")
    if task.run_id != deterministic_run_id(
        task.generator,
        task.dgp_seed,
        task.noise_level,
        task.optimizer_seed,
        task.radius,
        task.alpha,
    ):
        raise DevelopmentRunnerError("Task run ID is invalid.")

    m = seasonal_period(task.generator)
    L = frozen_lag_dimension(m)
    V = pc_nfpso_validation_size(PRETEST_LENGTH, L)

    try:
        series = generate_synthetic_series(
            task.generator,
            dgp_seed=task.dgp_seed,
            noise_level=task.noise_level,
        )
        observed = np.asarray(series.observed, dtype=float)
        if observed.shape != (SERIES_LENGTH,) or not np.all(np.isfinite(observed)):
            raise DevelopmentRunnerError("Synthetic generator returned invalid observations.")
        raw_pretest = observed[:PRETEST_LENGTH].copy()
        test_actuals = observed[PRETEST_LENGTH:].copy()
        if len(test_actuals) != TEST_LENGTH:
            raise DevelopmentRunnerError("Frozen synthetic train/test split length mismatch.")
    except Exception as exc:
        return _failure(task, "generation", exc, m=m, L=L, V=V)

    try:
        fitted = fit_function(
            raw_pretest,
            n_lags=L,
            validation_size=V,
            radius=task.radius,
            alpha=task.alpha,
            optimizer_seed=task.optimizer_seed,
        )
    except Exception as exc:
        return _failure(task, "fit", exc, m=m, L=L, V=V)

    try:
        predicted = np.asarray(
            forecast_function(fitted, test_actuals), dtype=float
        ).reshape(-1)
        if predicted.shape != test_actuals.shape or not np.all(np.isfinite(predicted)):
            raise DevelopmentRunnerError("PC-NFPSO forecast is invalid.")
    except Exception as exc:
        return _failure(task, "forecast", exc, m=m, L=L, V=V)

    try:
        scale = raw_mase_scale(raw_pretest, seasonal_period_value=m)
        mase = mase_from_predictions(test_actuals, predicted, mase_scale=scale)
        best_objective = float(fitted.optimizer_result.best_cost)
        n_rules = int(fitted.final_model.n_rules)
        if not np.isfinite(best_objective):
            raise DevelopmentRunnerError("Best objective is non-finite.")
        if n_rules < 1:
            raise DevelopmentRunnerError("Final model has no rules.")
    except Exception as exc:
        return _failure(task, "metrics", exc, m=m, L=L, V=V)

    return DevelopmentRunResult(
        task=task,
        status="success",
        mase=float(mase),
        mase_scale=float(scale),
        n_lags=L,
        validation_size=V,
        seasonal_period=m,
        best_objective=best_objective,
        n_rules=n_rules,
        failure=None,
    )


def _failure_to_dict(failure: DevelopmentRunFailure | None) -> dict[str, str] | None:
    return None if failure is None else asdict(failure)


def _result_to_dict(result: DevelopmentRunResult) -> dict[str, object]:
    return {
        "task": _task_to_dict(result.task),
        "status": result.status,
        "mase": result.mase,
        "mase_scale": result.mase_scale,
        "n_lags": result.n_lags,
        "validation_size": result.validation_size,
        "seasonal_period": result.seasonal_period,
        "best_objective": result.best_objective,
        "n_rules": result.n_rules,
        "failure": _failure_to_dict(result.failure),
    }


def _result_from_dict(data: dict[str, object]) -> DevelopmentRunResult:
    task = DevelopmentRunTask(**data["task"])  # type: ignore[arg-type]
    failure_data = data.get("failure")
    failure = (
        None
        if failure_data is None
        else DevelopmentRunFailure(**failure_data)  # type: ignore[arg-type]
    )
    result = DevelopmentRunResult(
        task=task,
        status=str(data["status"]),  # type: ignore[arg-type]
        mase=None if data.get("mase") is None else float(data["mase"]),
        mase_scale=None if data.get("mase_scale") is None else float(data["mase_scale"]),
        n_lags=int(data["n_lags"]),
        validation_size=int(data["validation_size"]),
        seasonal_period=int(data["seasonal_period"]),
        best_objective=None if data.get("best_objective") is None else float(data["best_objective"]),
        n_rules=None if data.get("n_rules") is None else int(data["n_rules"]),
        failure=failure,
    )
    validate_development_result(result)
    return result


def validate_development_result(result: DevelopmentRunResult) -> None:
    task = result.task
    assert_development_seed_firewall(task.dgp_seed, task.optimizer_seed, task.noise_level)
    if task.run_id != deterministic_run_id(
        task.generator, task.dgp_seed, task.noise_level, task.optimizer_seed, task.radius, task.alpha
    ):
        raise DevelopmentRunnerError("Result contains invalid task run ID.")
    expected_m = seasonal_period(task.generator)
    expected_lag = frozen_lag_dimension(expected_m)
    expected_validation = pc_nfpso_validation_size(PRETEST_LENGTH, expected_lag)
    if result.seasonal_period != expected_m or result.n_lags != expected_lag or result.validation_size != expected_validation:
        raise DevelopmentRunnerError("Result protocol dimensions are inconsistent with the task.")

    if result.status == "success":
        if result.failure is not None:
            raise DevelopmentRunnerError("Successful result must not contain failure provenance.")
        if result.mase is None or not np.isfinite(result.mase) or result.mase < 0.0:
            raise DevelopmentRunnerError("Successful result must contain finite non-negative MASE.")
        if result.mase_scale is None or not np.isfinite(result.mase_scale) or result.mase_scale <= MIN_MASE_SCALE:
            raise DevelopmentRunnerError("Successful result must contain a valid MASE scale.")
        if result.best_objective is None or not np.isfinite(result.best_objective):
            raise DevelopmentRunnerError("Successful result must contain finite best objective.")
        if result.n_rules is None or result.n_rules < 1:
            raise DevelopmentRunnerError("Successful result must contain a positive rule count.")
    elif result.status == "failed":
        if result.failure is None:
            raise DevelopmentRunnerError("Failed result requires failure provenance.")
        if result.failure.stage not in {"generation", "fit", "forecast", "metrics"}:
            raise DevelopmentRunnerError("Invalid development failure stage.")
        if result.mase is not None or result.mase_scale is not None:
            raise DevelopmentRunnerError("Failed result must not contain an accuracy value or MASE scale.")
    else:
        raise DevelopmentRunnerError(f"Invalid development result status: {result.status!r}")


def result_path(output_dir: str | os.PathLike[str], run_id: str) -> Path:
    if not isinstance(run_id, str) or len(run_id) != 64:
        raise DevelopmentRunnerError("run_id must be a full SHA-256 hex string.")
    return Path(output_dir) / "results" / f"{run_id}.json"


def write_development_result(
    output_dir: str | os.PathLike[str],
    result: DevelopmentRunResult,
) -> Path:
    validate_development_result(result)
    path = result_path(output_dir, result.task.run_id)
    if path.exists():
        raise DevelopmentRunnerError(f"Result already exists for run {result.task.run_id}.")
    payload = _canonical_json_bytes(_result_to_dict(result))
    _atomic_write_bytes(path, payload)
    return path


def load_development_result(path: str | os.PathLike[str]) -> DevelopmentRunResult:
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DevelopmentRunnerError(f"Invalid development result file: {p}") from exc
    return _result_from_dict(data)


def collect_development_results(
    output_dir: str | os.PathLike[str],
    manifest: DevelopmentManifest,
) -> tuple[DevelopmentRunResult, ...]:
    results: list[DevelopmentRunResult] = []
    for task in manifest.tasks:
        path = result_path(output_dir, task.run_id)
        if not path.is_file():
            continue
        result = load_development_result(path)
        if result.task != task:
            raise DevelopmentRunnerError(
                f"Stored result task does not match manifest for run {task.run_id}."
            )
        results.append(result)
    return tuple(results)


def pending_development_tasks(
    output_dir: str | os.PathLike[str],
    manifest: DevelopmentManifest,
) -> tuple[DevelopmentRunTask, ...]:
    pending = []
    for task in manifest.tasks:
        path = result_path(output_dir, task.run_id)
        if not path.exists():
            pending.append(task)
            continue
        result = load_development_result(path)
        if result.task != task:
            raise DevelopmentRunnerError(
                f"Stored result task does not match manifest for run {task.run_id}."
            )
    return tuple(pending)


def run_pending_development_tasks(
    output_dir: str | os.PathLike[str],
    manifest: DevelopmentManifest,
    *,
    max_runs: int | None = None,
    execute_function: Callable[[DevelopmentRunTask], DevelopmentRunResult] = execute_development_task,
) -> tuple[DevelopmentRunResult, ...]:
    pending = pending_development_tasks(output_dir, manifest)
    if max_runs is not None:
        if isinstance(max_runs, bool) or not isinstance(max_runs, int) or max_runs < 0:
            raise DevelopmentRunnerError("max_runs must be a non-negative integer or None.")
        pending = pending[:max_runs]
    completed: list[DevelopmentRunResult] = []
    for task in pending:
        result = execute_function(task)
        if result.task != task:
            raise DevelopmentRunnerError("Execution function returned a result for the wrong task.")
        validate_development_result(result)
        write_development_result(output_dir, result)
        completed.append(result)
    return tuple(completed)


def selection_from_complete_development_results(
    results: Iterable[DevelopmentRunResult],
):
    rows = tuple(results)
    if len(rows) != EXPECTED_RUNS:
        raise DevelopmentRunnerError(
            f"Selection requires exactly {EXPECTED_RUNS} completed runs; got {len(rows)}."
        )
    run_ids = {result.task.run_id for result in rows}
    if len(run_ids) != EXPECTED_RUNS:
        raise DevelopmentRunnerError("Selection input contains duplicate run IDs.")
    failed = [result for result in rows if not result.successful]
    if failed:
        first = failed[0]
        raise DevelopmentRunnerError(
            "A007 blocks radius-alpha selection when any development run failed; "
            f"failed_count={len(failed)}, first_run={first.task.run_id}, "
            f"stage={first.failure.stage if first.failure else 'unknown'}."
        )
    scores = tuple(result.to_score() for result in rows)
    try:
        return select_global_radius_alpha(scores, expected_generators=ALL_GENERATORS)
    except DevelopmentSelectionError as exc:
        raise DevelopmentRunnerError("Development selection gate rejected the collected scores.") from exc


__all__ = [
    "SCHEMA_VERSION",
    "PRETEST_LENGTH",
    "TEST_LENGTH",
    "EXPECTED_CONDITIONS",
    "EXPECTED_RUNS",
    "PROTOCOL_INPUT_FILES",
    "DevelopmentRunnerError",
    "DevelopmentRunTask",
    "DevelopmentRunFailure",
    "DevelopmentRunResult",
    "DevelopmentManifest",
    "DevelopmentDryRunSummary",
    "frozen_lag_dimension",
    "pc_nfpso_validation_size",
    "raw_mase_scale",
    "mase_from_predictions",
    "deterministic_run_id",
    "build_development_tasks",
    "validate_development_tasks",
    "development_dry_run_summary",
    "protocol_fingerprint",
    "git_head",
    "assert_clean_git_worktree",
    "build_development_manifest",
    "write_development_manifest",
    "load_development_manifest",
    "prepare_development_workspace",
    "execute_development_task",
    "validate_development_result",
    "result_path",
    "write_development_result",
    "load_development_result",
    "collect_development_results",
    "pending_development_tasks",
    "run_pending_development_tasks",
    "selection_from_complete_development_results",
]
