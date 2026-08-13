from __future__ import annotations

import hashlib

import numpy as np
import pytest

from nfpsosc.v2.synthetic import (
    ALL_GENERATORS,
    CANONICAL_GENERATORS,
    CONFIRMATORY_NOISE_LEVELS,
    CUSTOM_GENERATORS,
    SERIES_LENGTH,
    apply_relative_observation_noise,
    generate_clean_series,
    generate_noise_bundle,
    generate_synthetic_series,
    henon_initial_condition,
    list_generators,
    seasonal_period,
)


def test_exact_nine_generator_manifest() -> None:
    assert CUSTOM_GENERATORS == (
        "linear_ar",
        "nonlinear_sine",
        "regime_switching",
        "spiky_outbreak",
        "noisy_seasonal",
        "short_memory",
    )
    assert CANONICAL_GENERATORS == ("logistic_map", "henon_map", "lorenz63")
    assert list_generators() == ALL_GENERATORS
    assert len(ALL_GENERATORS) == 9
    assert len(set(ALL_GENERATORS)) == 9


@pytest.mark.parametrize("generator", ALL_GENERATORS)
def test_all_generators_return_finite_length_180(generator: str) -> None:
    s = generate_synthetic_series(generator, dgp_seed=1001, noise_level=0.10)
    assert s.clean.shape == (SERIES_LENGTH,)
    assert s.observed.shape == (SERIES_LENGTH,)
    assert np.all(np.isfinite(s.clean))
    assert np.all(np.isfinite(s.observed))


@pytest.mark.parametrize("generator", ALL_GENERATORS)
def test_reproducibility_same_seed(generator: str) -> None:
    a = generate_synthetic_series(generator, dgp_seed=1007, noise_level=0.10)
    b = generate_synthetic_series(generator, dgp_seed=1007, noise_level=0.10)
    assert np.array_equal(a.clean, b.clean)
    assert np.array_equal(a.observed, b.observed)


@pytest.mark.parametrize("generator", ALL_GENERATORS)
def test_positive_noise_differs_across_dgp_seeds(generator: str) -> None:
    a = generate_synthetic_series(generator, dgp_seed=1001, noise_level=0.10)
    b = generate_synthetic_series(generator, dgp_seed=1002, noise_level=0.10)
    assert not np.array_equal(a.observed, b.observed)


@pytest.mark.parametrize("generator", ALL_GENERATORS)
def test_zero_noise_is_exact_clean_copy(generator: str) -> None:
    s = generate_synthetic_series(generator, dgp_seed=1003, noise_level=0.0)
    assert np.array_equal(s.observed, s.clean)
    assert s.observed is not s.clean


def test_negative_noise_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        generate_synthetic_series("linear_ar", dgp_seed=1001, noise_level=-0.01)


def test_population_std_ddof_zero_and_seeded_noise_exact() -> None:
    clean = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float64)
    level = 0.2
    seed = 8123
    minimum = 1.0
    z = np.random.default_rng(seed).normal(size=len(clean))
    expected = clean + level * max(float(np.std(clean, ddof=0)), minimum) * z
    got = apply_relative_observation_noise(
        clean, noise_level=level, noise_seed=seed, minimum_scale=minimum
    )
    assert np.array_equal(got, expected)


@pytest.mark.parametrize(
    "generator",
    [g for g in ALL_GENERATORS if g != "spiky_outbreak"],
)
def test_same_standard_normal_vector_is_shared_across_positive_noise_levels(generator: str) -> None:
    bundle = generate_noise_bundle(generator, dgp_seed=1005, noise_levels=(0.05, 0.10, 0.20))
    clean = bundle[0.05].clean
    z05 = (bundle[0.05].observed - clean) / 0.05
    z10 = (bundle[0.10].observed - clean) / 0.10
    z20 = (bundle[0.20].observed - clean) / 0.20
    assert np.allclose(z05, z10, rtol=0.0, atol=2e-13)
    assert np.allclose(z05, z20, rtol=0.0, atol=2e-13)


def test_noise_bundle_reuses_identical_clean_trajectory() -> None:
    bundle = generate_noise_bundle("lorenz63", dgp_seed=1001)
    assert tuple(bundle) == CONFIRMATORY_NOISE_LEVELS
    assert np.array_equal(bundle[0.05].clean, bundle[0.10].clean)
    assert np.array_equal(bundle[0.10].clean, bundle[0.20].clean)


def test_spiky_outbreak_is_nonnegative_before_and_after_contamination() -> None:
    clean = generate_clean_series("spiky_outbreak", dgp_seed=1001)
    noisy = generate_synthetic_series("spiky_outbreak", dgp_seed=1001, noise_level=0.20)
    assert np.min(clean) >= 0.0
    assert np.min(noisy.observed) >= 0.0


def test_deterministic_custom_clean_signals_ignore_dgp_seed() -> None:
    for generator in ("nonlinear_sine", "noisy_seasonal"):
        a = generate_clean_series(generator, dgp_seed=1001)
        b = generate_clean_series(generator, dgp_seed=1009)
        assert np.array_equal(a, b)


def test_custom_clean_equations_exact_at_selected_points() -> None:
    t = np.arange(SERIES_LENGTH, dtype=np.float64)
    s = np.sin(2.0 * np.pi * t / 24.0)
    expected_nonlinear = 10.0 * s + 3.5 * s * s + 0.03 * t
    assert np.array_equal(generate_clean_series("nonlinear_sine", dgp_seed=1001), expected_nonlinear)

    expected_seasonal = (
        50.0
        + 0.04 * t
        + 8.0 * np.sin(2.0 * np.pi * t / 12.0)
        + 2.8 * np.cos(4.0 * np.pi * t / 12.0)
    )
    assert np.array_equal(generate_clean_series("noisy_seasonal", dgp_seed=1001), expected_seasonal)


def test_logistic_burn_in_and_first_stride_regression() -> None:
    y = generate_clean_series("logistic_map", dgp_seed=1001)
    assert y[0] == pytest.approx(0.09794285907028949, rel=0.0, abs=1e-15)
    assert y[1] == pytest.approx(0.3534002217097076, rel=0.0, abs=1e-15)


def test_henon_burn_in_and_first_stride_regression() -> None:
    y = generate_clean_series("henon_map", dgp_seed=1001)
    assert y[0] == pytest.approx(0.42571744856521265, rel=0.0, abs=1e-15)
    assert y[1] == pytest.approx(0.5052617439120684, rel=0.0, abs=1e-15)


def test_lorenz_frozen_rk4_order_burn_in_and_stride_regression() -> None:
    y = generate_clean_series("lorenz63", dgp_seed=1001)
    assert y[0] == pytest.approx(13.762481521700717, rel=0.0, abs=1e-12)
    assert y[1] == pytest.approx(11.170882547675332, rel=0.0, abs=1e-12)


def test_frozen_seasonal_period_metadata() -> None:
    assert seasonal_period("nonlinear_sine") == 24
    assert seasonal_period("noisy_seasonal") == 12
    for generator in ALL_GENERATORS:
        if generator not in {"nonlinear_sine", "noisy_seasonal"}:
            assert seasonal_period(generator) == 1


def test_unknown_generator_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown V2 synthetic generator"):
        generate_synthetic_series("unknown", dgp_seed=1001, noise_level=0.10)


def test_duplicate_bundle_noise_levels_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        generate_noise_bundle("linear_ar", dgp_seed=1001, noise_levels=(0.1, 0.1))


def test_returned_arrays_are_read_only() -> None:
    s = generate_synthetic_series("linear_ar", dgp_seed=1001, noise_level=0.10)
    assert not s.clean.flags.writeable
    assert not s.observed.flags.writeable


_HENON_DEVELOPMENT_PRE_A008_SHA256 = {
    1001: "b7754e49ca0a7b4fcb6f73121d483f37dcc512ce8c739bdc4bcfe70601322428",
    1002: "29b73b7fc94cf35911a4c3d958fb0850ee064b89dbdfd0c62f748e5894666af0",
    1003: "1fb7906cf9e40136eacec533b3d67ddae6bd25cf3917f2f7a721ee3f04968d1a",
    1004: "b3cab392ef06b4146a52d384fb5d12c75fccd21e71afc8bdf3435bf459b57636",
    1005: "c876869e934968f95ef2dcf5badb5061c85bd86c36c2b5e1278458daf91d6d7f",
    1006: "82114bf4f48956cac1de5529e399813df3f719698795e0df4ff3f2f0a94877f2",
    1007: "e6ca0a85545c7510549f500f03b110b9be6d43f3d78dadc9ce2a7c8ee868cce8",
    1008: "4e568da35fd82f5d46cf80ba328bf8a8d39c854c8ccdaf24bc5cf67227c73a53",
    1009: "c27b50f5a4cf7052bd7141fd3644e78c9a5cead0cb0d96857b5af291a80e04c6",
    1010: "46ce0ba93ec4dfd4961b465d73763f61c6d6b745f64f3da6eb93db05d539ed39",
}


@pytest.mark.parametrize(
    "dgp_seed,expected_sha256",
    sorted(_HENON_DEVELOPMENT_PRE_A008_SHA256.items()),
)
def test_a008_preserves_all_henon_development_trajectories_bit_for_bit(
    dgp_seed: int, expected_sha256: str
) -> None:
    x0, y0, candidate_number = henon_initial_condition(dgp_seed)
    assert candidate_number == 1
    assert -0.5 <= x0 <= 0.5
    assert -0.5 <= y0 <= 0.5
    y = generate_clean_series("henon_map", dgp_seed=dgp_seed)
    payload = np.asarray(y, dtype="<f8").tobytes(order="C")
    assert hashlib.sha256(payload).hexdigest() == expected_sha256


def test_a008_debug_seed_1_accepts_candidate_three_exactly() -> None:
    x0, y0, candidate_number = henon_initial_condition(1)
    assert candidate_number == 3
    assert x0 == pytest.approx(-0.18816854798951455, rel=0.0, abs=0.0)
    assert y0 == pytest.approx(-0.07667355102742435, rel=0.0, abs=0.0)
    y = generate_clean_series("henon_map", dgp_seed=1)
    assert y.shape == (SERIES_LENGTH,)
    assert np.all(np.isfinite(y))


def test_a008_final_seed_2003_accepts_candidate_three_exactly() -> None:
    x0, y0, candidate_number = henon_initial_condition(2003)
    assert candidate_number == 3
    assert x0 == pytest.approx(0.48212728096900015, rel=0.0, abs=0.0)
    assert y0 == pytest.approx(-0.3580221771573131, rel=0.0, abs=0.0)
    y = generate_clean_series("henon_map", dgp_seed=2003)
    assert y.shape == (SERIES_LENGTH,)
    assert np.all(np.isfinite(y))


def test_a008_all_frozen_dgp_seed_domains_are_finite_length_180() -> None:
    seed_domains = (1,) + tuple(range(1001, 1011)) + tuple(range(2001, 2021))
    checked = 0
    for generator in ALL_GENERATORS:
        for dgp_seed in seed_domains:
            y = generate_clean_series(generator, dgp_seed=dgp_seed)
            assert y.shape == (SERIES_LENGTH,)
            assert np.all(np.isfinite(y))
            checked += 1
    assert checked == 279
