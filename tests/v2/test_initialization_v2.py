import inspect

import numpy as np
import pytest

from nfpsosc.v2.initialization import (
    FEATURE_RANGE_FLOOR,
    SCALER_OUT_MAX,
    SCALER_OUT_MIN,
    SC_WIDTH_DENOMINATOR,
    FrozenMinMaxScaler1D,
    feature_ranges,
    initialize_tsk_from_subtractive_clustering,
    make_antecedent_bounds,
    subtractive_clustering,
)


def _fit_data():
    X = np.array(
        [
            [0.10, 0.20],
            [0.20, 0.30],
            [0.30, 0.40],
            [0.70, 0.65],
            [0.80, 0.75],
            [0.90, 0.85],
        ],
        dtype=float,
    )
    y = np.array([0.15, 0.22, 0.31, 0.66, 0.76, 0.88], dtype=float)
    return X, y


def test_scaler_fits_exact_fitting_extrema_and_fixed_interval():
    scaler = FrozenMinMaxScaler1D.fit(np.array([10.0, 20.0, 15.0]))
    assert scaler.data_min == 10.0
    assert scaler.data_max == 20.0
    assert scaler.out_min == SCALER_OUT_MIN
    assert scaler.out_max == SCALER_OUT_MAX
    got = scaler.transform(np.array([10.0, 20.0]))
    np.testing.assert_allclose(got, [0.1, 0.9], rtol=0.0, atol=1e-15)


def test_scaler_is_frozen_validation_extremes_do_not_refit_or_clip():
    fitting = np.array([10.0, 12.0, 20.0])
    scaler = FrozenMinMaxScaler1D.fit(fitting)
    validation = np.array([-100.0, 100.0])
    transformed = scaler.transform(validation)
    assert scaler.data_min == 10.0
    assert scaler.data_max == 20.0
    assert transformed[0] < 0.1
    assert transformed[1] > 0.9


def test_scaler_round_trip():
    scaler = FrozenMinMaxScaler1D.fit(np.array([-3.0, 7.0, 2.0]))
    values = np.array([-5.0, -3.0, 0.0, 7.0, 12.0])
    np.testing.assert_allclose(
        scaler.inverse_transform(scaler.transform(values)),
        values,
        rtol=0.0,
        atol=1e-13,
    )


def test_scaler_rejects_constant_fitting_segment():
    with pytest.raises(ValueError, match="strictly positive range"):
        FrozenMinMaxScaler1D.fit(np.ones(8))


def test_subtractive_clustering_is_deterministic_and_non_mutating():
    X, y = _fit_data()
    joint = np.column_stack([X, y])
    before = joint.copy()
    a = subtractive_clustering(joint, radius=0.55)
    b = subtractive_clustering(joint, radius=0.55)
    np.testing.assert_array_equal(joint, before)
    np.testing.assert_allclose(a.centers, b.centers, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(a.potentials, b.potentials, rtol=0.0, atol=0.0)


def test_initialization_clusters_joint_X_y_and_retains_input_center_dimensions():
    X, y = _fit_data()
    init = initialize_tsk_from_subtractive_clustering(X, y, radius=0.55)
    assert init.clustering.centers.shape[1] == X.shape[1] + 1
    np.testing.assert_allclose(
        init.model.centers,
        init.clustering.centers[:, : X.shape[1]],
        rtol=0.0,
        atol=0.0,
    )


def test_initialization_api_has_no_validation_or_test_arguments():
    names = set(inspect.signature(initialize_tsk_from_subtractive_clustering).parameters)
    assert names == {"X_fit", "y_fit", "radius"}
    assert not any("validation" in name or "test" in name for name in names)


def test_initial_width_formula_exactly_matches_a004():
    X, y = _fit_data()
    radius = 0.35
    init = initialize_tsk_from_subtractive_clustering(X, y, radius=radius)
    expected = radius * feature_ranges(X) / SC_WIDTH_DENOMINATOR
    np.testing.assert_allclose(
        init.model.sigmas,
        np.tile(expected, (init.model.n_rules, 1)),
        rtol=1e-14,
        atol=1e-14,
    )


def test_feature_range_has_frozen_1e_minus_3_floor():
    X = np.array([[2.0, 1.0], [2.0, 1.0001], [2.0, 1.0002]])
    got = feature_ranges(X)
    np.testing.assert_allclose(got, [FEATURE_RANGE_FLOOR, FEATURE_RANGE_FLOOR])


def test_antecedent_bounds_exactly_match_frozen_formula_and_order():
    X, _ = _fit_data()
    bounds = make_antecedent_bounds(X, n_rules=2)
    r = feature_ranges(X)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    expected_lower = np.concatenate(
        [np.tile(lo - 0.05 * r, 2), np.tile(0.05 * r, 2)]
    )
    expected_upper = np.concatenate(
        [np.tile(hi + 0.05 * r, 2), np.tile(1.00 * r, 2)]
    )
    np.testing.assert_allclose(bounds.lower, expected_lower)
    np.testing.assert_allclose(bounds.upper, expected_upper)


def test_particle0_is_clipped_to_frozen_bounds_and_has_expected_order():
    X, y = _fit_data()
    init = initialize_tsk_from_subtractive_clustering(X, y, radius=1.0)
    expected_unclipped = np.concatenate(
        [init.model.centers.ravel(), init.model.sigmas.ravel()]
    )
    np.testing.assert_allclose(init.particle0_unclipped, expected_unclipped)
    np.testing.assert_allclose(
        init.particle0,
        np.clip(expected_unclipped, init.bounds.lower, init.bounds.upper),
    )
    assert np.all(init.particle0 >= init.bounds.lower)
    assert np.all(init.particle0 <= init.bounds.upper)


def test_initialization_consequents_are_zero_not_silently_ridge_fitted():
    X, y = _fit_data()
    init = initialize_tsk_from_subtractive_clustering(X, y, radius=0.55)
    np.testing.assert_array_equal(init.model.consequents, 0.0)


def test_changing_fit_targets_can_change_joint_clustering_but_not_feature_width_formula():
    X, y = _fit_data()
    y2 = y.copy()
    y2[:3] += 10.0
    a = initialize_tsk_from_subtractive_clustering(X, y, radius=0.35)
    b = initialize_tsk_from_subtractive_clustering(X, y2, radius=0.35)
    expected = 0.35 * feature_ranges(X) / SC_WIDTH_DENOMINATOR
    np.testing.assert_allclose(a.model.sigmas[0], expected)
    np.testing.assert_allclose(b.model.sigmas[0], expected)
    # Joint clustering stores the target dimension explicitly; therefore the
    # clustering inputs/results are target-aware rather than X-only.
    assert a.clustering.centers.shape[1] == X.shape[1] + 1
    assert b.clustering.centers.shape[1] == X.shape[1] + 1


def test_invalid_radius_is_rejected_before_initialization():
    X, y = _fit_data()
    for radius in (0.0, -0.5, np.inf, np.nan):
        with pytest.raises(ValueError, match="radius"):
            initialize_tsk_from_subtractive_clustering(X, y, radius=radius)
