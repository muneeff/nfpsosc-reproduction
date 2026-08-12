from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np


FROZEN_RADII: tuple[float, ...] = (0.25, 0.35, 0.55, 0.75, 1.0)
FROZEN_ALPHAS: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1)
LEGACY_V1_DGP_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
DEVELOPMENT_DGP_SEEDS: tuple[int, ...] = tuple(range(1001, 1011))
FINAL_DGP_SEEDS: tuple[int, ...] = tuple(range(2001, 2021))
DEVELOPMENT_OPTIMIZER_SEEDS: tuple[int, ...] = (31001, 31002, 31003)
FINAL_OPTIMIZER_SEEDS: tuple[int, ...] = (41001, 41002, 41003, 41004, 41005)
CONFIRMATORY_NOISE_LEVELS: tuple[float, ...] = (0.05, 0.10, 0.20)
DIAGNOSTIC_NOISE_LEVEL: float = 0.0


class DevelopmentSelectionError(ValueError):
    """Raised when a development-selection input violates the frozen V2 protocol."""


@dataclass(frozen=True, order=True)
class CandidatePair:
    radius: float
    alpha: float


@dataclass(frozen=True)
class DevelopmentScore:
    generator: str
    dgp_seed: int
    noise_level: float
    optimizer_seed: int
    radius: float
    alpha: float
    mase: float

    @property
    def pair(self) -> CandidatePair:
        return CandidatePair(float(self.radius), float(self.alpha))

    @property
    def condition(self) -> tuple[str, int, float]:
        return (self.generator, int(self.dgp_seed), float(self.noise_level))


@dataclass(frozen=True)
class PairSelectionSummary:
    pair: CandidatePair
    selection_statistic: float
    condition_count: int


@dataclass(frozen=True)
class DevelopmentSelectionResult:
    selected_pair: CandidatePair
    summaries: tuple[PairSelectionSummary, ...]
    condition_count: int
    record_count: int
    generator_count: int


def frozen_candidate_grid() -> tuple[CandidatePair, ...]:
    """Return the frozen Cartesian radius-alpha grid in deterministic order."""
    return tuple(CandidatePair(float(r), float(a)) for r, a in product(FROZEN_RADII, FROZEN_ALPHAS))


def assert_development_seed_firewall(
    dgp_seed: int,
    optimizer_seed: int,
    noise_level: float,
) -> None:
    """Reject any seed/noise value outside the locked development domain."""
    dgp_seed = int(dgp_seed)
    optimizer_seed = int(optimizer_seed)
    noise_level = float(noise_level)

    if dgp_seed not in DEVELOPMENT_DGP_SEEDS:
        if dgp_seed in LEGACY_V1_DGP_SEEDS:
            reason = "legacy V1 DGP seed"
        elif dgp_seed in FINAL_DGP_SEEDS:
            reason = "final locked DGP seed"
        else:
            reason = "non-development DGP seed"
        raise DevelopmentSelectionError(f"Development selection rejected {reason}: {dgp_seed}.")

    if optimizer_seed not in DEVELOPMENT_OPTIMIZER_SEEDS:
        if optimizer_seed in FINAL_OPTIMIZER_SEEDS:
            reason = "final locked optimizer seed"
        else:
            reason = "non-development optimizer seed"
        raise DevelopmentSelectionError(
            f"Development selection rejected {reason}: {optimizer_seed}."
        )

    if noise_level not in CONFIRMATORY_NOISE_LEVELS:
        if noise_level == DIAGNOSTIC_NOISE_LEVEL:
            reason = "noise-free diagnostic level"
        else:
            reason = "non-confirmatory noise level"
        raise DevelopmentSelectionError(
            f"Development selection rejected {reason}: {noise_level}."
        )


def _validate_record(record: DevelopmentScore) -> None:
    if not isinstance(record.generator, str) or not record.generator.strip():
        raise DevelopmentSelectionError("generator must be a non-empty string.")

    assert_development_seed_firewall(
        record.dgp_seed,
        record.optimizer_seed,
        record.noise_level,
    )

    if record.pair not in set(frozen_candidate_grid()):
        raise DevelopmentSelectionError(
            f"Candidate pair is outside the frozen grid: {record.pair}."
        )

    mase = float(record.mase)
    if not np.isfinite(mase) or mase < 0.0:
        raise DevelopmentSelectionError(
            f"MASE must be finite and non-negative; received {record.mase!r}."
        )


def _condition_sort_key(condition: tuple[str, int, float]) -> tuple[str, int, float]:
    return (condition[0], condition[1], condition[2])


def select_global_radius_alpha(
    records: Iterable[DevelopmentScore],
    *,
    expected_generators: Sequence[str] | None = None,
) -> DevelopmentSelectionResult:
    """Select the single global V2 radius-alpha pair from complete development scores.

    Protocol order:
    1. validate the development seed/noise firewall;
    2. require all 35 frozen candidate pairs and identical complete conditions;
    3. aggregate the three optimizer replicates by median within each condition;
    4. aggregate condition values by their median for each candidate pair;
    5. minimize that statistic, breaking exact ties by larger radius then larger alpha.
    """

    rows = tuple(records)
    if not rows:
        raise DevelopmentSelectionError("Development selection requires at least one score.")

    grid = frozen_candidate_grid()
    grid_set = set(grid)
    development_optimizer_set = set(DEVELOPMENT_OPTIMIZER_SEEDS)

    grouped: dict[
        CandidatePair,
        dict[tuple[str, int, float], dict[int, float]],
    ] = {}
    observed_generators: set[str] = set()
    observed_dgp_seeds: set[int] = set()
    observed_noise_levels: set[float] = set()

    for row in rows:
        _validate_record(row)
        observed_generators.add(row.generator)
        observed_dgp_seeds.add(int(row.dgp_seed))
        observed_noise_levels.add(float(row.noise_level))

        by_condition = grouped.setdefault(row.pair, {})
        by_seed = by_condition.setdefault(row.condition, {})
        seed = int(row.optimizer_seed)
        if seed in by_seed:
            raise DevelopmentSelectionError(
                "Duplicate development score for "
                f"pair={row.pair}, condition={row.condition}, optimizer_seed={seed}."
            )
        by_seed[seed] = float(row.mase)

    if set(grouped) != grid_set:
        missing = tuple(pair for pair in grid if pair not in grouped)
        extra = tuple(pair for pair in grouped if pair not in grid_set)
        raise DevelopmentSelectionError(
            f"Candidate-grid coverage mismatch; missing={missing}, extra={extra}."
        )

    if observed_dgp_seeds != set(DEVELOPMENT_DGP_SEEDS):
        raise DevelopmentSelectionError(
            "Development DGP seed coverage must equal the frozen 1001-1010 set; "
            f"observed={sorted(observed_dgp_seeds)}."
        )

    if observed_noise_levels != set(CONFIRMATORY_NOISE_LEVELS):
        raise DevelopmentSelectionError(
            "Noise-level coverage must equal the frozen confirmatory set "
            f"{CONFIRMATORY_NOISE_LEVELS}; observed={sorted(observed_noise_levels)}."
        )

    if expected_generators is not None:
        expected = {str(g) for g in expected_generators}
        if "" in expected or not expected:
            raise DevelopmentSelectionError(
                "expected_generators must contain non-empty generator names."
            )
        if observed_generators != expected:
            raise DevelopmentSelectionError(
                "Generator coverage mismatch; "
                f"expected={sorted(expected)}, observed={sorted(observed_generators)}."
            )

    expected_conditions = {
        (generator, dgp_seed, noise_level)
        for generator in observed_generators
        for dgp_seed in DEVELOPMENT_DGP_SEEDS
        for noise_level in CONFIRMATORY_NOISE_LEVELS
    }

    common_conditions: set[tuple[str, int, float]] | None = None
    for pair in grid:
        conditions = set(grouped[pair])
        if conditions != expected_conditions:
            missing = sorted(expected_conditions - conditions, key=_condition_sort_key)
            extra = sorted(conditions - expected_conditions, key=_condition_sort_key)
            raise DevelopmentSelectionError(
                f"Condition coverage mismatch for {pair}; missing={missing[:5]}, "
                f"extra={extra[:5]}."
            )
        if common_conditions is None:
            common_conditions = conditions
        elif conditions != common_conditions:
            raise DevelopmentSelectionError(
                f"Candidate {pair} does not share identical development conditions."
            )

        for condition in expected_conditions:
            observed_optimizer_seeds = set(grouped[pair][condition])
            if observed_optimizer_seeds != development_optimizer_set:
                raise DevelopmentSelectionError(
                    "Optimizer replicate coverage mismatch for "
                    f"pair={pair}, condition={condition}; "
                    f"expected={sorted(development_optimizer_set)}, "
                    f"observed={sorted(observed_optimizer_seeds)}."
                )

    condition_order = tuple(sorted(expected_conditions, key=_condition_sort_key))
    summaries: list[PairSelectionSummary] = []
    for pair in grid:
        condition_medians = []
        for condition in condition_order:
            replicate_values = [
                grouped[pair][condition][seed]
                for seed in DEVELOPMENT_OPTIMIZER_SEEDS
            ]
            condition_medians.append(float(np.median(replicate_values)))

        selection_statistic = float(np.median(condition_medians))
        if not np.isfinite(selection_statistic):
            raise DevelopmentSelectionError(
                f"Non-finite selection statistic for candidate {pair}."
            )
        summaries.append(
            PairSelectionSummary(
                pair=pair,
                selection_statistic=selection_statistic,
                condition_count=len(condition_order),
            )
        )

    # Exact ties only: the frozen amendment specifies larger radius, then larger alpha.
    selected = min(
        summaries,
        key=lambda summary: (
            summary.selection_statistic,
            -summary.pair.radius,
            -summary.pair.alpha,
        ),
    )

    return DevelopmentSelectionResult(
        selected_pair=selected.pair,
        summaries=tuple(summaries),
        condition_count=len(condition_order),
        record_count=len(rows),
        generator_count=len(observed_generators),
    )
