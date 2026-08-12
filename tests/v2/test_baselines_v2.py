import numpy as np
import pandas as pd
import pytest

from nfpsosc.v2.baselines import (
    FrozenTheta,
    first_forecast_value,
)


def make_theta_series() -> pd.Series:
    t = np.arange(48, dtype=float)
    values = (
        10.0
        + 0.2 * t
        + 2.0 * np.sin(2.0 * np.pi * t / 12.0)
    )

    # Deliberately non-zero index.
    return pd.Series(
        values,
        index=pd.RangeIndex(100, 148),
    )


def test_position_based_forecast_extraction_nonzero_index():
    from statsmodels.tsa.forecasting.theta import ThetaModel

    y = make_theta_series()

    result = ThetaModel(
        y,
        period=12,
        deseasonalize=True,
        use_test=False,
        method="additive",
    ).fit()

    forecast = result.forecast(1, theta=2.0)

    assert forecast.index[0] == 148

    # Correct extraction.
    actual = first_forecast_value(forecast)

    assert actual == pytest.approx(
        float(forecast.iloc[0]),
        rel=0.0,
        abs=0.0,
    )

    # Reproduce the exact V1 failure mode.
    with pytest.raises(KeyError):
        _ = forecast[0]


def test_frozen_theta_first_forecast_matches_statsmodels_exactly():
    from statsmodels.tsa.forecasting.theta import ThetaModel

    y = make_theta_series()

    sm_result = ThetaModel(
        y,
        period=12,
        deseasonalize=True,
        use_test=False,
        method="additive",
    ).fit()

    frozen = FrozenTheta.fit(
        y,
        seasonal_period=12,
    )

    expected = float(
        sm_result.forecast(
            1,
            theta=2.0,
        ).iloc[0]
    )

    actual = frozen.forecast_one()

    assert actual == pytest.approx(
        expected,
        rel=0.0,
        abs=1e-12,
    )


def test_frozen_theta_parameters_do_not_change_after_observation():
    y = make_theta_series()

    frozen = FrozenTheta.fit(
        y,
        seasonal_period=12,
    )

    alpha_before = frozen.alpha
    b0_before = frozen.b0

    prediction = frozen.forecast_one()

    frozen.observe(
        prediction + 0.75
    )

    assert frozen.alpha == alpha_before
    assert frozen.b0 == b0_before


def test_frozen_theta_updates_ses_state_from_observed_truth():
    y = make_theta_series()

    frozen = FrozenTheta.fit(
        y,
        seasonal_period=12,
    )

    old_ses = frozen.ses_state
    alpha = frozen.alpha
    seasonal = frozen.current_seasonal()

    observed = frozen.forecast_one() + 1.25

    expected_deseasonalized = observed - seasonal
    expected_new_ses = (
        alpha * expected_deseasonalized
        + (1.0 - alpha) * old_ses
    )

    frozen.observe(observed)

    assert frozen.ses_state == pytest.approx(
        expected_new_ses,
        rel=0.0,
        abs=1e-12,
    )


def test_frozen_theta_advances_nobs_without_refitting_parameters():
    y = make_theta_series()

    frozen = FrozenTheta.fit(
        y,
        seasonal_period=12,
    )

    n_before = frozen.nobs
    alpha_before = frozen.alpha
    b0_before = frozen.b0

    observed = frozen.forecast_one()

    frozen.observe(observed)

    assert frozen.nobs == n_before + 1
    assert frozen.alpha == alpha_before
    assert frozen.b0 == b0_before


def test_frozen_theta_seasonal_cycle_advances():
    y = make_theta_series()

    frozen = FrozenTheta.fit(
        y,
        seasonal_period=12,
    )

    first = frozen.current_seasonal()

    frozen.observe(
        frozen.forecast_one()
    )

    second = frozen.current_seasonal()

    assert first == pytest.approx(
        0.0,
        abs=1e-10,
    )

    assert second == pytest.approx(
        1.0,
        abs=1e-10,
    )


def test_nonseasonal_theta_uses_zero_seasonal_component():
    t = np.arange(40, dtype=float)
    y = pd.Series(
        5.0 + 0.1 * t,
        index=pd.RangeIndex(50, 90),
    )

    frozen = FrozenTheta.fit(
        y,
        seasonal_period=1,
    )

    assert frozen.current_seasonal() == pytest.approx(
        0.0,
        abs=0.0,
    )

    pred = frozen.forecast_one()

    assert np.isfinite(pred)


def test_theta_rejects_nonfinite_observation():
    y = make_theta_series()

    frozen = FrozenTheta.fit(
        y,
        seasonal_period=12,
    )

    with pytest.raises(
        ValueError,
        match="observed value must be finite",
    ):
        frozen.observe(np.nan)

def make_ar_series(n=80):
    rng = np.random.default_rng(12345)

    y = np.zeros(n, dtype=float)
    y[0] = 1.0

    for t in range(1, n):
        y[t] = (
            0.65 * y[t - 1]
            + rng.normal(0.0, 0.25)
        )

    return y


def test_frozen_sarima_forecast_is_finite():
    from nfpsosc.v2.baselines import FrozenSARIMA

    y = make_ar_series()

    model = FrozenSARIMA.fit(
        y,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
    )

    pred = model.forecast_one()

    assert np.isfinite(pred)


def test_frozen_sarima_parameters_do_not_change_after_observation():
    from nfpsosc.v2.baselines import FrozenSARIMA

    y = make_ar_series()

    model = FrozenSARIMA.fit(
        y,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
    )

    params_before = model.params.copy()

    observed = model.forecast_one() + 2.0
    model.observe(observed)

    np.testing.assert_array_equal(
        model.params,
        params_before,
    )


def test_frozen_sarima_observed_truth_updates_state():
    from nfpsosc.v2.baselines import FrozenSARIMA

    y = make_ar_series()

    updated = FrozenSARIMA.fit(
        y,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
    )

    static = FrozenSARIMA.fit(
        y,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
    )

    first = updated.forecast_one()

    # Give the updated model a deliberately surprising true observation.
    updated.observe(first + 3.0)

    next_updated = updated.forecast_one()

    # Static model receives no observed test truth.
    next_static = static.forecast_one()

    assert np.isfinite(next_updated)
    assert np.isfinite(next_static)

    # State update must matter.
    assert next_updated != pytest.approx(
        next_static,
        rel=0.0,
        abs=1e-10,
    )


def test_frozen_sarima_observation_count_advances():
    from nfpsosc.v2.baselines import FrozenSARIMA

    y = make_ar_series()

    model = FrozenSARIMA.fit(
        y,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
    )

    before = model.nobs

    model.observe(
        model.forecast_one()
    )

    assert model.nobs == before + 1


def test_frozen_sarima_rejects_nonfinite_observation():
    from nfpsosc.v2.baselines import FrozenSARIMA

    y = make_ar_series()

    model = FrozenSARIMA.fit(
        y,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
    )

    with pytest.raises(
        ValueError,
        match="observed value must be finite",
    ):
        model.observe(np.nan)


def test_frozen_sarima_uses_no_test_time_refit(monkeypatch):
    from nfpsosc.v2.baselines import FrozenSARIMA

    y = make_ar_series()

    model = FrozenSARIMA.fit(
        y,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
    )

    original_append = model._result.append
    calls = []

    def checked_append(endog, refit=False, **kwargs):
        calls.append(refit)

        return original_append(
            endog,
            refit=refit,
            **kwargs,
        )

    monkeypatch.setattr(
        model._result,
        "append",
        checked_append,
    )

    model.observe(
        model.forecast_one()
    )

    assert calls == [False]
def make_ets_series(n=72):
    rng = np.random.default_rng(24680)
    t = np.arange(n, dtype=float)

    return (
        20.0
        + 0.15 * t
        + 3.0 * np.sin(2.0 * np.pi * t / 12.0)
        + rng.normal(0.0, 0.15, size=n)
    )


def test_frozen_ets_first_forecast_matches_statsmodels():
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from nfpsosc.v2.baselines import FrozenETS

    y = make_ets_series()

    direct = ExponentialSmoothing(
        y,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
        use_boxcox=False,
    ).fit(
        optimized=True,
        remove_bias=False,
    )

    frozen = FrozenETS.fit(
        y,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=12,
    )

    expected = float(
        np.asarray(direct.forecast(1)).reshape(-1)[0]
    )

    actual = frozen.forecast_one()

    assert actual == pytest.approx(
        expected,
        rel=0.0,
        abs=1e-10,
    )


def test_frozen_ets_parameters_do_not_change_after_observation():
    from nfpsosc.v2.baselines import FrozenETS

    y = make_ets_series()

    model = FrozenETS.fit(
        y,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=12,
    )

    params_before = model.frozen_parameters()

    model.observe(
        model.forecast_one() + 2.0
    )

    params_after = model.frozen_parameters()

    assert params_after.keys() == params_before.keys()

    for key in params_before:
        before = params_before[key]
        after = params_after[key]

        if isinstance(before, np.ndarray):
            np.testing.assert_array_equal(
                after,
                before,
            )
        else:
            assert after == before


def test_frozen_ets_observed_truth_changes_future_state():
    from nfpsosc.v2.baselines import FrozenETS

    y = make_ets_series()

    updated = FrozenETS.fit(
        y,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=12,
    )

    static = FrozenETS.fit(
        y,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=12,
    )

    first = updated.forecast_one()

    updated.observe(first + 5.0)

    next_updated = updated.forecast_one()
    next_static = static.forecast_one()

    assert np.isfinite(next_updated)
    assert np.isfinite(next_static)

    assert next_updated != pytest.approx(
        next_static,
        rel=0.0,
        abs=1e-10,
    )


def test_frozen_ets_replay_uses_optimized_false():
    from nfpsosc.v2.baselines import FrozenETS

    y = make_ets_series()

    model = FrozenETS.fit(
        y,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=12,
    )

    replay = model._replay_result()

    # statsmodels can encode the fixed-parameter mask as None
    # instead of literal False.  What matters is that no
    # optimizer was run and no parameter was marked optimized.
    assert replay.mle_retvals is None
    assert not bool(np.any(replay.optimized))

    assert replay.params["smoothing_level"] == pytest.approx(
        model.smoothing_level,
        rel=0.0,
        abs=0.0,
    )

    assert replay.params["smoothing_trend"] == pytest.approx(
        model.smoothing_trend,
        rel=0.0,
        abs=0.0,
    )

    assert replay.params["smoothing_seasonal"] == pytest.approx(
        model.smoothing_seasonal,
        rel=0.0,
        abs=0.0,
    )

    assert replay.params["damping_trend"] == pytest.approx(
        model.damping_trend,
        rel=0.0,
        abs=0.0,
    )

def test_frozen_ets_observation_count_advances():
    from nfpsosc.v2.baselines import FrozenETS

    y = make_ets_series()

    model = FrozenETS.fit(
        y,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=12,
    )

    before = model.nobs

    model.observe(
        model.forecast_one()
    )

    assert model.nobs == before + 1


def test_frozen_ets_rejects_nonfinite_observation():
    from nfpsosc.v2.baselines import FrozenETS

    y = make_ets_series()

    model = FrozenETS.fit(
        y,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=12,
    )

    with pytest.raises(
        ValueError,
        match="observed value must be finite",
    ):
        model.observe(np.nan)


def test_frozen_ets_nonseasonal_configuration():
    from nfpsosc.v2.baselines import FrozenETS

    t = np.arange(50, dtype=float)
    y = 5.0 + 0.2 * t

    model = FrozenETS.fit(
        y,
        trend=None,
        damped_trend=False,
        seasonal=None,
        seasonal_periods=None,
    )

    assert np.isfinite(
        model.forecast_one()
    )
def make_lag_regression_series(n=90):
    rng = np.random.default_rng(98765)

    y = np.zeros(n, dtype=float)
    y[:2] = [1.0, 0.5]

    for t in range(2, n):
        y[t] = (
            0.55 * y[t - 1]
            - 0.20 * y[t - 2]
            + rng.normal(0.0, 0.15)
        )

    return y


def test_inner_validation_size_matches_frozen_formula():
    from nfpsosc.v2.baselines import inner_validation_size

    # V=min(max(5,ceil(.20*N_train)), N_train-L-10)
    assert inner_validation_size(80, 5) == 16
    assert inner_validation_size(40, 5) == 8
    assert inner_validation_size(25, 5) == 5

    with pytest.raises(ValueError):
        inner_validation_size(19, 5)


def test_frozen_ridge_fit_once_and_observed_history_updates():
    from nfpsosc.v2.baselines import FrozenLagRegressor

    y = make_lag_regression_series()

    model = FrozenLagRegressor.fit_ridge(
        y,
        n_lags=5,
        alpha=0.1,
    )

    estimator_id = id(model.estimator)

    first = model.forecast_one()
    model.observe(first + 2.0)
    second = model.forecast_one()

    assert id(model.estimator) == estimator_id
    assert np.isfinite(first)
    assert np.isfinite(second)

    # New observed truth entered the lag vector.
    assert second != pytest.approx(first)


def test_frozen_svr_fit_once_and_observed_history_updates():
    from nfpsosc.v2.baselines import FrozenLagRegressor

    y = make_lag_regression_series()

    model = FrozenLagRegressor.fit_svr(
        y,
        n_lags=5,
        C=10.0,
        epsilon=0.05,
        gamma="scale",
    )

    estimator_id = id(model.estimator)

    first = model.forecast_one()
    model.observe(first + 2.0)
    second = model.forecast_one()

    assert id(model.estimator) == estimator_id
    assert np.isfinite(first)
    assert np.isfinite(second)


def test_frozen_xgboost_fit_once_and_observed_history_updates():
    from nfpsosc.v2.baselines import FrozenLagRegressor

    y = make_lag_regression_series()

    model = FrozenLagRegressor.fit_xgboost(
        y,
        n_lags=5,
        n_estimators=100,
        max_depth=2,
        learning_rate=0.03,
        min_child_weight=1,
    )

    estimator_id = id(model.estimator)

    first = model.forecast_one()
    model.observe(first + 2.0)
    second = model.forecast_one()

    assert id(model.estimator) == estimator_id
    assert np.isfinite(first)
    assert np.isfinite(second)


def test_lag_model_rejects_nonfinite_observation():
    from nfpsosc.v2.baselines import FrozenLagRegressor

    y = make_lag_regression_series()

    model = FrozenLagRegressor.fit_ridge(
        y,
        n_lags=5,
        alpha=1.0,
    )

    with pytest.raises(
        ValueError,
        match="observed value must be finite",
    ):
        model.observe(np.nan)


def test_ridge_tuner_uses_frozen_candidate_grid():
    from nfpsosc.v2.baselines import select_ridge

    y = make_lag_regression_series()

    selected = select_ridge(
        y,
        n_lags=5,
        seasonal_period=1,
        alphas=[
            1e-6,
            1e-4,
            1e-2,
            0.1,
            1.0,
            10.0,
            100.0,
        ],
    )

    assert selected.alpha in {
        1e-6,
        1e-4,
        1e-2,
        0.1,
        1.0,
        10.0,
        100.0,
    }

    assert np.isfinite(selected.validation_mase)


def test_svr_tuner_returns_frozen_grid_member():
    from nfpsosc.v2.baselines import select_svr

    y = make_lag_regression_series()

    selected = select_svr(
        y,
        n_lags=5,
        seasonal_period=1,
        C_values=[0.1, 1.0, 10.0, 100.0],
        epsilon_values=[0.01, 0.05, 0.1],
        gamma_values=["scale", 0.1, 1.0],
    )

    assert selected.C in {0.1, 1.0, 10.0, 100.0}
    assert selected.epsilon in {0.01, 0.05, 0.1}
    assert selected.gamma in {"scale", 0.1, 1.0}
    assert np.isfinite(selected.validation_mase)


def test_xgboost_tuner_returns_frozen_grid_member():
    from nfpsosc.v2.baselines import select_xgboost

    y = make_lag_regression_series()

    selected = select_xgboost(
        y,
        n_lags=5,
        seasonal_period=1,
        n_estimators_values=[100, 300],
        max_depth_values=[2, 3],
        learning_rate_values=[0.03, 0.1],
        min_child_weight_values=[1, 5],
    )

    assert selected.n_estimators in {100, 300}
    assert selected.max_depth in {2, 3}
    assert selected.learning_rate in {0.03, 0.1}
    assert selected.min_child_weight in {1, 5}
    assert np.isfinite(selected.validation_mase)
def test_ridge_has_zero_fit_calls_during_test_window(monkeypatch):
    from sklearn.linear_model import Ridge
    from nfpsosc.v2.baselines import FrozenLagRegressor

    y = make_lag_regression_series()

    model = FrozenLagRegressor.fit_ridge(
        y,
        n_lags=5,
        alpha=0.1,
    )

    def forbidden_fit(*args, **kwargs):
        raise AssertionError(
            "Ridge.fit() called during frozen test window"
        )

    monkeypatch.setattr(
        Ridge,
        "fit",
        forbidden_fit,
    )

    for _ in range(5):
        pred = model.forecast_one()

        assert np.isfinite(pred)

        model.observe(
            pred + 0.25
        )


def test_svr_has_zero_fit_calls_during_test_window(monkeypatch):
    from sklearn.svm import SVR
    from nfpsosc.v2.baselines import FrozenLagRegressor

    y = make_lag_regression_series()

    model = FrozenLagRegressor.fit_svr(
        y,
        n_lags=5,
        C=10.0,
        epsilon=0.05,
        gamma="scale",
    )

    def forbidden_fit(*args, **kwargs):
        raise AssertionError(
            "SVR.fit() called during frozen test window"
        )

    monkeypatch.setattr(
        SVR,
        "fit",
        forbidden_fit,
    )

    for _ in range(5):
        pred = model.forecast_one()

        assert np.isfinite(pred)

        model.observe(
            pred + 0.25
        )


def test_xgboost_has_zero_fit_calls_during_test_window(monkeypatch):
    from xgboost import XGBRegressor
    from nfpsosc.v2.baselines import FrozenLagRegressor

    y = make_lag_regression_series()

    model = FrozenLagRegressor.fit_xgboost(
        y,
        n_lags=5,
        n_estimators=100,
        max_depth=2,
        learning_rate=0.03,
        min_child_weight=1,
    )

    def forbidden_fit(*args, **kwargs):
        raise AssertionError(
            "XGBRegressor.fit() called during frozen test window"
        )

    monkeypatch.setattr(
        XGBRegressor,
        "fit",
        forbidden_fit,
    )

    for _ in range(5):
        pred = model.forecast_one()

        assert np.isfinite(pred)

        model.observe(
            pred + 0.25
        )
def test_sarima_candidates_nonseasonal_are_exactly_frozen_grid():
    from nfpsosc.v2.baselines import generate_sarima_candidates

    candidates = generate_sarima_candidates(
        seasonal_period=1,
    )

    # 9 ARMA combinations *:
    # d=0 -> no-constant + constant = 18
    # d=1 -> no-constant only = 9
    assert len(candidates) == 27

    for candidate in candidates:
        p, d, q = candidate.order
        P, D, Q, m = candidate.seasonal_order

        assert p in {0, 1, 2}
        assert d in {0, 1}
        assert q in {0, 1, 2}

        assert (P, D, Q, m) == (0, 0, 0, 0)

        assert p + q <= 4

        if candidate.trend == "c":
            assert d == 0
        else:
            assert candidate.trend is None


def test_sarima_candidates_seasonal_obey_all_frozen_constraints():
    from nfpsosc.v2.baselines import generate_sarima_candidates

    candidates = generate_sarima_candidates(
        seasonal_period=12,
    )

    assert len(candidates) == 155

    for candidate in candidates:
        p, d, q = candidate.order
        P, D, Q, m = candidate.seasonal_order

        assert p in {0, 1, 2}
        assert d in {0, 1}
        assert q in {0, 1, 2}

        assert P in {0, 1}
        assert D in {0, 1}
        assert Q in {0, 1}
        assert m == 12

        assert p + q + P + Q <= 4

        if candidate.trend == "c":
            assert d == 0
            assert D == 0


def test_sarima_tie_break_prefers_frozen_complexity_order():
    from nfpsosc.v2.baselines import (
        SARIMACandidate,
        sarima_tie_key,
    )

    simpler = SARIMACandidate(
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 12),
        trend=None,
    )

    more_complex = SARIMACandidate(
        order=(1, 0, 1),
        seasonal_order=(0, 0, 0, 12),
        trend=None,
    )

    assert sarima_tie_key(
        simpler
    ) < sarima_tie_key(
        more_complex
    )


def test_sarima_exact_tie_prefers_no_constant_last():
    from nfpsosc.v2.baselines import (
        SARIMACandidate,
        sarima_tie_key,
    )

    no_constant = SARIMACandidate(
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 12),
        trend=None,
    )

    constant = SARIMACandidate(
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 12),
        trend="c",
    )

    assert sarima_tie_key(
        no_constant
    ) < sarima_tie_key(
        constant
    )


def test_ets_candidates_nonseasonal_are_exactly_three():
    from nfpsosc.v2.baselines import generate_ets_candidates

    candidates = generate_ets_candidates(
        seasonal_period=1,
    )

    assert len(candidates) == 3

    structures = {
        (
            c.trend,
            c.damped_trend,
            c.seasonal,
            c.seasonal_periods,
        )
        for c in candidates
    }

    assert structures == {
        (None, False, None, None),
        ("add", False, None, None),
        ("add", True, None, None),
    }


def test_ets_candidates_seasonal_are_exactly_six():
    from nfpsosc.v2.baselines import generate_ets_candidates

    candidates = generate_ets_candidates(
        seasonal_period=12,
    )

    assert len(candidates) == 6

    for candidate in candidates:
        assert candidate.trend in {
            None,
            "add",
        }

        assert candidate.seasonal in {
            None,
            "add",
        }

        if candidate.trend is None:
            assert candidate.damped_trend is False

        if candidate.seasonal is None:
            assert candidate.seasonal_periods is None
        else:
            assert candidate.seasonal_periods == 12


def test_ets_tie_break_prefers_simpler_structure():
    from nfpsosc.v2.baselines import (
        ETSCandidate,
        ets_tie_key,
    )

    simplest = ETSCandidate(
        trend=None,
        damped_trend=False,
        seasonal=None,
        seasonal_periods=None,
    )

    trend_only = ETSCandidate(
        trend="add",
        damped_trend=False,
        seasonal=None,
        seasonal_periods=None,
    )

    trend_and_season = ETSCandidate(
        trend="add",
        damped_trend=False,
        seasonal="add",
        seasonal_periods=12,
    )

    assert ets_tie_key(
        simplest
    ) < ets_tie_key(
        trend_only
    )

    assert ets_tie_key(
        trend_only
    ) < ets_tie_key(
        trend_and_season
    )


def test_sarima_selector_skips_failure_and_logs_it(monkeypatch):
    import types
    import nfpsosc.v2.baselines as baselines

    y = make_ar_series()

    calls = []

    def fake_fit(training_series, candidate):
        calls.append(candidate)

        if candidate == baselines.generate_sarima_candidates(1)[0]:
            raise RuntimeError("synthetic candidate failure")

        score = (
            1.0
            if candidate.order == (1, 0, 0)
            and candidate.trend is None
            else 10.0
        )

        return types.SimpleNamespace(
            aicc=score,
            converged=True,
        )

    monkeypatch.setattr(
        baselines,
        "_fit_sarima_candidate",
        fake_fit,
    )

    selected = baselines.select_sarima_aicc(
        y,
        seasonal_period=1,
    )

    assert selected.candidate.order == (
        1,
        0,
        0,
    )
    assert selected.candidate.trend is None
    assert selected.aicc == pytest.approx(1.0)

    assert len(selected.failures) == 1
    assert (
        selected.failures[0].error_type
        == "RuntimeError"
    )

    assert "synthetic candidate failure" in (
        selected.failures[0].message
    )


def test_ets_selector_uses_tie_break_after_equal_aicc(monkeypatch):
    import types
    import nfpsosc.v2.baselines as baselines

    y = make_ets_series()

    def fake_fit(training_series, candidate):
        return types.SimpleNamespace(
            aicc=5.0,
            optimizer_success=True,
        )

    monkeypatch.setattr(
        baselines,
        "_fit_ets_candidate",
        fake_fit,
    )

    selected = baselines.select_ets_aicc(
        y,
        seasonal_period=12,
    )

    # All AICc values are exactly tied:
    # Amendment V2-A001 must select simplest ETS.
    assert selected.candidate.trend is None
    assert selected.candidate.damped_trend is False
    assert selected.candidate.seasonal is None
    assert selected.candidate.seasonal_periods is None


def test_sarima_all_candidates_fail_without_fallback(monkeypatch):
    import nfpsosc.v2.baselines as baselines

    y = make_ar_series()

    def always_fail(training_series, candidate):
        raise RuntimeError(
            "forced SARIMA failure"
        )

    monkeypatch.setattr(
        baselines,
        "_fit_sarima_candidate",
        always_fail,
    )

    with pytest.raises(
        baselines.BaselineSelectionError,
        match="no fallback permitted",
    ) as exc_info:
        baselines.select_sarima_aicc(
            y,
            seasonal_period=1,
        )

    assert len(
        exc_info.value.failures
    ) == len(
        baselines.generate_sarima_candidates(1)
    )


def test_ets_all_candidates_fail_without_fallback(monkeypatch):
    import nfpsosc.v2.baselines as baselines

    y = make_ets_series()

    def always_fail(training_series, candidate):
        raise RuntimeError(
            "forced ETS failure"
        )

    monkeypatch.setattr(
        baselines,
        "_fit_ets_candidate",
        always_fail,
    )

    with pytest.raises(
        baselines.BaselineSelectionError,
        match="no fallback permitted",
    ) as exc_info:
        baselines.select_ets_aicc(
            y,
            seasonal_period=12,
        )

    assert len(
        exc_info.value.failures
    ) == len(
        baselines.generate_ets_candidates(12)
    )
def test_sarima_nonconverged_best_aicc_cannot_win(monkeypatch):
    import types
    import nfpsosc.v2.baselines as baselines

    y = make_ar_series()

    candidates = baselines.generate_sarima_candidates(1)
    bad = candidates[0]
    good = candidates[1]

    def fake_fit(training_series, candidate):
        if candidate == bad:
            # Deliberately best AICc, but invalid numerically.
            return types.SimpleNamespace(
                aicc=-1000.0,
                converged=False,
            )

        if candidate == good:
            return types.SimpleNamespace(
                aicc=1.0,
                converged=True,
            )

        return types.SimpleNamespace(
            aicc=10.0,
            converged=True,
        )

    monkeypatch.setattr(
        baselines,
        "_fit_sarima_candidate",
        fake_fit,
    )

    selected = baselines.select_sarima_aicc(
        y,
        seasonal_period=1,
    )

    assert selected.candidate == good
    assert selected.aicc == pytest.approx(1.0)

    failures = [
        failure
        for failure in selected.failures
        if failure.candidate == bad
    ]

    assert len(failures) == 1

    assert (
        failures[0].error_type
        == "CandidateValidationError"
    )

    assert "did not converge" in failures[0].message


def test_ets_unsuccessful_optimizer_best_aicc_cannot_win(
    monkeypatch,
):
    import types
    import nfpsosc.v2.baselines as baselines

    y = make_ets_series()

    candidates = baselines.generate_ets_candidates(12)
    bad = candidates[0]
    good = candidates[1]

    def fake_fit(training_series, candidate):
        if candidate == bad:
            # Deliberately best AICc, but optimizer failed.
            return types.SimpleNamespace(
                aicc=-1000.0,
                optimizer_success=False,
            )

        if candidate == good:
            return types.SimpleNamespace(
                aicc=1.0,
                optimizer_success=True,
            )

        return types.SimpleNamespace(
            aicc=10.0,
            optimizer_success=True,
        )

    monkeypatch.setattr(
        baselines,
        "_fit_ets_candidate",
        fake_fit,
    )

    selected = baselines.select_ets_aicc(
        y,
        seasonal_period=12,
    )

    assert selected.candidate == good
    assert selected.aicc == pytest.approx(1.0)

    failures = [
        failure
        for failure in selected.failures
        if failure.candidate == bad
    ]

    assert len(failures) == 1

    assert (
        failures[0].error_type
        == "CandidateValidationError"
    )

    assert "optimizer did not converge" in failures[0].message