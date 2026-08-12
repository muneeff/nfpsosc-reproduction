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