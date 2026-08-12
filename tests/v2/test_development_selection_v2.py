from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from nfpsosc.v2.development_selection import (
    CONFIRMATORY_NOISE_LEVELS,
    DEVELOPMENT_DGP_SEEDS,
    DEVELOPMENT_OPTIMIZER_SEEDS,
    FINAL_DGP_SEEDS,
    FINAL_OPTIMIZER_SEEDS,
    FROZEN_ALPHAS,
    FROZEN_RADII,
    LEGACY_V1_DGP_SEEDS,
    CandidatePair,
    DevelopmentScore,
    DevelopmentSelectionError,
    assert_development_seed_firewall,
    frozen_candidate_grid,
    select_global_radius_alpha,
)


def _records(*, generators=("g1",), score_fn=None):
    if score_fn is None:
        score_fn = lambda pair, condition, opt_seed: 5.0 + pair.radius + pair.alpha
    rows = []
    for pair in frozen_candidate_grid():
        for generator in generators:
            for dgp_seed in DEVELOPMENT_DGP_SEEDS:
                for noise_level in CONFIRMATORY_NOISE_LEVELS:
                    condition = (generator, dgp_seed, noise_level)
                    for optimizer_seed in DEVELOPMENT_OPTIMIZER_SEEDS:
                        rows.append(
                            DevelopmentScore(
                                generator=generator,
                                dgp_seed=dgp_seed,
                                noise_level=noise_level,
                                optimizer_seed=optimizer_seed,
                                radius=pair.radius,
                                alpha=pair.alpha,
                                mase=float(score_fn(pair, condition, optimizer_seed)),
                            )
                        )
    return rows


def test_frozen_candidate_grid_is_exact_cartesian_product_in_deterministic_order():
    grid = frozen_candidate_grid()
    assert len(grid) == 35
    assert grid[0] == CandidatePair(0.25, 1e-6)
    assert grid[-1] == CandidatePair(1.0, 0.1)
    assert tuple(sorted({p.radius for p in grid})) == FROZEN_RADII
    assert tuple(sorted({p.alpha for p in grid})) == FROZEN_ALPHAS
    assert len(set(grid)) == 35


def test_seed_firewall_accepts_only_development_domain_example():
    assert_development_seed_firewall(1001, 31001, 0.05)
    assert_development_seed_firewall(1010, 31003, 0.20)


def test_seed_firewall_rejects_legacy_dgp_seed():
    with pytest.raises(DevelopmentSelectionError, match="legacy V1 DGP seed"):
        assert_development_seed_firewall(LEGACY_V1_DGP_SEEDS[0], 31001, 0.05)


def test_seed_firewall_rejects_final_dgp_seed():
    with pytest.raises(DevelopmentSelectionError, match="final locked DGP seed"):
        assert_development_seed_firewall(FINAL_DGP_SEEDS[0], 31001, 0.05)


def test_seed_firewall_rejects_final_optimizer_seed():
    with pytest.raises(DevelopmentSelectionError, match="final locked optimizer seed"):
        assert_development_seed_firewall(1001, FINAL_OPTIMIZER_SEEDS[0], 0.05)


def test_seed_firewall_rejects_unknown_optimizer_seed():
    with pytest.raises(DevelopmentSelectionError, match="non-development optimizer seed"):
        assert_development_seed_firewall(1001, 99999, 0.05)


def test_seed_firewall_rejects_zero_noise_diagnostic_level():
    with pytest.raises(DevelopmentSelectionError, match="noise-free diagnostic level"):
        assert_development_seed_firewall(1001, 31001, 0.0)


def test_selection_requires_all_35_candidate_pairs():
    rows = _records()
    missing_pair = frozen_candidate_grid()[0]
    rows = [r for r in rows if r.pair != missing_pair]
    with pytest.raises(DevelopmentSelectionError, match="Candidate-grid coverage mismatch"):
        select_global_radius_alpha(rows)


def test_selection_rejects_duplicate_optimizer_record():
    rows = _records()
    rows.append(rows[0])
    with pytest.raises(DevelopmentSelectionError, match="Duplicate development score"):
        select_global_radius_alpha(rows)


def test_selection_rejects_missing_optimizer_replicate():
    rows = _records()
    rows.pop(0)
    with pytest.raises(DevelopmentSelectionError, match="Optimizer replicate coverage mismatch"):
        select_global_radius_alpha(rows)


def test_selection_requires_complete_seed_noise_cartesian_conditions():
    rows = _records()
    victim_pair = frozen_candidate_grid()[0]
    rows = [
        r
        for r in rows
        if not (
            r.pair == victim_pair
            and r.dgp_seed == DEVELOPMENT_DGP_SEEDS[-1]
            and r.noise_level == CONFIRMATORY_NOISE_LEVELS[-1]
        )
    ]
    with pytest.raises(DevelopmentSelectionError, match="Condition coverage mismatch"):
        select_global_radius_alpha(rows)


def test_selection_rejects_nonfinite_mase():
    rows = _records()
    rows[0] = replace(rows[0], mase=np.nan)
    with pytest.raises(DevelopmentSelectionError, match="finite and non-negative"):
        select_global_radius_alpha(rows)


def test_selection_rejects_negative_mase():
    rows = _records()
    rows[0] = replace(rows[0], mase=-1.0)
    with pytest.raises(DevelopmentSelectionError, match="finite and non-negative"):
        select_global_radius_alpha(rows)


def test_expected_generator_firewall_rejects_mismatch():
    rows = _records(generators=("g1",))
    with pytest.raises(DevelopmentSelectionError, match="Generator coverage mismatch"):
        select_global_radius_alpha(rows, expected_generators=("g1", "g2"))


def test_optimizer_replicates_are_aggregated_by_median_before_condition_median():
    target = CandidatePair(0.25, 1e-6)

    def score_fn(pair, condition, opt_seed):
        if pair == target:
            return {31001: 1.0, 31002: 100.0, 31003: 2.0}[opt_seed]
        return 10.0

    result = select_global_radius_alpha(_records(score_fn=score_fn))
    summary = next(s for s in result.summaries if s.pair == target)
    assert summary.selection_statistic == pytest.approx(2.0)
    assert result.selected_pair == target


def test_global_selection_uses_median_across_development_conditions():
    target = CandidatePair(0.55, 0.01)

    def score_fn(pair, condition, opt_seed):
        generator, dgp_seed, noise = condition
        if pair == target:
            return 1.0 + 0.001 * (dgp_seed - 1001) + noise
        return 3.0 + pair.radius + pair.alpha

    result = select_global_radius_alpha(_records(score_fn=score_fn))
    assert result.selected_pair == target
    assert result.condition_count == 30
    assert result.record_count == 35 * 30 * 3
    assert result.generator_count == 1


def test_exact_full_tie_chooses_larger_radius_then_larger_alpha():
    result = select_global_radius_alpha(
        _records(score_fn=lambda pair, condition, opt_seed: 1.0)
    )
    assert result.selected_pair == CandidatePair(1.0, 0.1)


def test_exact_alpha_tie_with_same_radius_chooses_larger_alpha():
    a = CandidatePair(0.55, 0.05)
    b = CandidatePair(0.55, 0.1)

    def score_fn(pair, condition, opt_seed):
        if pair in {a, b}:
            return 1.0
        return 2.0

    result = select_global_radius_alpha(_records(score_fn=score_fn))
    assert result.selected_pair == b


def test_summaries_preserve_frozen_candidate_order():
    result = select_global_radius_alpha(_records())
    assert tuple(summary.pair for summary in result.summaries) == frozen_candidate_grid()
