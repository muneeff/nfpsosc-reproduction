from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import nfpsosc.v2.pc_nfpso as integration
from nfpsosc.v2.initialization import FrozenMinMaxScaler1D
from nfpsosc.v2.model import TSKModel
from nfpsosc.v2.pc_nfpso import (
    FROZEN_ITERATIONS,
    FROZEN_PARTICLES,
    PCNFPSOFit,
    PCNFPSOTrainingData,
    fit_pc_nfpso_v2,
    forecast_pc_nfpso_v2,
    make_chronological_lags,
    prepare_pc_nfpso_training_data,
)


def _raw_series(n: int = 48) -> np.ndarray:
    t = np.arange(n, dtype=float)
    return 20.0 + 0.04 * t + 1.2 * np.sin(2.0 * np.pi * t / 12.0) + 0.1 * np.cos(0.7 * t)


def _fake_optimizer(monkeypatch, *, inspect_call=None):
    def fake(cost_function, initial_position, lower_bound, upper_bound, config):
        if inspect_call is not None:
            inspect_call(cost_function, initial_position, lower_bound, upper_bound, config)
        x = np.asarray(initial_position, dtype=float).copy()
        cost = float(cost_function(x))
        return SimpleNamespace(
            best_position=x,
            best_cost=cost,
            history={"best_cost": np.asarray([cost])},
        )

    monkeypatch.setattr(integration, "particle_swarm_optimize", fake)


def test_chronological_lags_have_frozen_raw_order():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X, target = make_chronological_lags(y, n_lags=3)
    np.testing.assert_array_equal(X, [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    np.testing.assert_array_equal(target, [4.0, 5.0])


def test_training_api_has_no_test_outcome_argument():
    names = set(inspect.signature(fit_pc_nfpso_v2).parameters)
    assert "raw_pretest" in names
    forbidden = {"raw_test", "test", "test_actuals", "y_test", "X_test"}
    assert names.isdisjoint(forbidden)


def test_scaler_fit_ends_at_final_fitting_target_and_excludes_validation_targets():
    raw = _raw_series(50)
    L = 5
    V = 8
    prepared = prepare_pc_nfpso_training_data(raw, n_lags=L, validation_size=V)
    n_fit = (len(raw) - L) - V
    assert prepared.fitting_raw_end == L + n_fit
    np.testing.assert_allclose(
        [prepared.scaler.data_min, prepared.scaler.data_max],
        [raw[: prepared.fitting_raw_end].min(), raw[: prepared.fitting_raw_end].max()],
    )


def test_validation_target_changes_do_not_change_scaler_or_fit_arrays():
    raw1 = _raw_series(50)
    raw2 = raw1.copy()
    V = 8
    raw2[-V:] += np.linspace(100.0, 800.0, V)
    a = prepare_pc_nfpso_training_data(raw1, n_lags=5, validation_size=V)
    b = prepare_pc_nfpso_training_data(raw2, n_lags=5, validation_size=V)
    assert a.scaler == b.scaler
    np.testing.assert_array_equal(a.X_fit, b.X_fit)
    np.testing.assert_array_equal(a.y_fit, b.y_fit)
    assert not np.array_equal(a.y_validation, b.y_validation)


def test_optimizer_receives_exact_frozen_budget_seed_bounds_and_particle0(monkeypatch):
    seen = {}

    def inspect_call(cost, initial, lower, upper, config):
        seen["iterations"] = config.iterations
        seen["particles"] = config.particles
        seen["seed"] = config.seed
        seen["dynamics"] = config.dynamics
        seen["boundary"] = config.boundary
        seen["initial"] = np.asarray(initial).copy()
        seen["lower"] = np.asarray(lower).copy()
        seen["upper"] = np.asarray(upper).copy()

    _fake_optimizer(monkeypatch, inspect_call=inspect_call)
    fit = fit_pc_nfpso_v2(
        _raw_series(),
        n_lags=5,
        validation_size=8,
        radius=0.55,
        alpha=1e-4,
        optimizer_seed=31001,
    )
    assert seen["iterations"] == FROZEN_ITERATIONS == 40
    assert seen["particles"] == FROZEN_PARTICLES == 12
    assert seen["seed"] == 31001
    assert seen["dynamics"] == "constricted"
    assert seen["boundary"] == "project"
    np.testing.assert_array_equal(seen["initial"], fit.initialization.particle0)
    np.testing.assert_array_equal(seen["lower"], fit.initialization.bounds.lower)
    np.testing.assert_array_equal(seen["upper"], fit.initialization.bounds.upper)


def test_optimizer_cost_uses_fit_and_validation_but_no_external_test_data(monkeypatch):
    calls = []
    original = integration.evaluate_antecedent_candidate

    def wrapped(template, vector, Xf, yf, Xv, yv, **kwargs):
        calls.append((Xf.copy(), yf.copy(), Xv.copy(), yv.copy()))
        return original(template, vector, Xf, yf, Xv, yv, **kwargs)

    monkeypatch.setattr(integration, "evaluate_antecedent_candidate", wrapped)
    _fake_optimizer(monkeypatch)
    fit = fit_pc_nfpso_v2(
        _raw_series(),
        n_lags=5,
        validation_size=8,
        radius=0.55,
        alpha=1e-4,
        optimizer_seed=31001,
    )
    assert len(calls) >= 2
    for Xf, yf, Xv, yv in calls:
        np.testing.assert_array_equal(Xf, fit.training.X_fit)
        np.testing.assert_array_equal(yf, fit.training.y_fit)
        np.testing.assert_array_equal(Xv, fit.training.X_validation)
        np.testing.assert_array_equal(yv, fit.training.y_validation)


def test_selected_candidate_consequents_are_fit_only(monkeypatch):
    _fake_optimizer(monkeypatch)
    fit = fit_pc_nfpso_v2(
        _raw_series(),
        n_lags=5,
        validation_size=8,
        radius=0.55,
        alpha=1e-4,
        optimizer_seed=31001,
    )
    expected = fit.initialization.model.with_antecedents(
        fit.optimizer_result.best_position
    )
    expected.fit_consequents(
        fit.training.X_fit,
        fit.training.y_fit,
        alpha=fit.alpha,
    )
    np.testing.assert_allclose(
        fit.selected_candidate.model.consequents,
        expected.consequents,
        rtol=1e-12,
        atol=1e-12,
    )


def test_final_refit_changes_only_consequents_and_uses_complete_pretest(monkeypatch):
    _fake_optimizer(monkeypatch)
    fit = fit_pc_nfpso_v2(
        _raw_series(),
        n_lags=5,
        validation_size=8,
        radius=0.55,
        alpha=1e-4,
        optimizer_seed=31001,
    )
    best = np.asarray(fit.optimizer_result.best_position)
    np.testing.assert_array_equal(fit.final_model.antecedent_vector(), best)

    expected = fit.initialization.model.with_antecedents(best)
    expected.fit_consequents(
        fit.training.X_pretest,
        fit.training.y_pretest,
        alpha=fit.alpha,
    )
    np.testing.assert_allclose(
        fit.final_model.consequents,
        expected.consequents,
        rtol=1e-12,
        atol=1e-12,
    )


class _LastLagModel:
    def __init__(self, n_features):
        self.n_features = n_features
        self.fit_calls = 0

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, -1]

    def fit_consequents(self, *args, **kwargs):
        self.fit_calls += 1
        raise AssertionError("test forecasting must never refit parameters")


def _forecast_fixture():
    raw_pretest = np.array([0.0, 2.0, 4.0, 6.0])
    scaler = FrozenMinMaxScaler1D.fit(np.array([0.0, 10.0]))
    training = SimpleNamespace(
        raw_pretest=raw_pretest,
        scaler=scaler,
        n_lags=2,
    )
    model = _LastLagModel(2)
    fitted = SimpleNamespace(training=training, final_model=model)
    return fitted, model


def test_test_forecasting_uses_observed_truth_history_not_recursive_predictions():
    fitted, model = _forecast_fixture()
    actuals = np.array([9.0, 1.0, 8.0])
    pred = forecast_pc_nfpso_v2(fitted, actuals)
    np.testing.assert_allclose(pred, [6.0, 9.0, 1.0], atol=1e-12)
    assert model.fit_calls == 0


def test_changing_future_actuals_cannot_change_first_forecast_but_updates_next_origin():
    fitted_a, _ = _forecast_fixture()
    fitted_b, _ = _forecast_fixture()
    a = forecast_pc_nfpso_v2(fitted_a, np.array([9.0, 1.0]))
    b = forecast_pc_nfpso_v2(fitted_b, np.array([-3.0, 100.0]))
    assert a[0] == pytest.approx(b[0])
    assert a[1] == pytest.approx(9.0)
    assert b[1] == pytest.approx(-3.0)


def test_forecast_does_not_mutate_real_tsk_parameters(monkeypatch):
    _fake_optimizer(monkeypatch)
    fit = fit_pc_nfpso_v2(
        _raw_series(),
        n_lags=5,
        validation_size=8,
        radius=0.55,
        alpha=1e-4,
        optimizer_seed=31001,
    )
    centers = fit.final_model.centers.copy()
    sigmas = fit.final_model.sigmas.copy()
    consequents = fit.final_model.consequents.copy()
    test_actuals = _raw_series(5) + 3.0
    pred = forecast_pc_nfpso_v2(fit, test_actuals)
    assert len(pred) == len(test_actuals)
    assert np.all(np.isfinite(pred))
    np.testing.assert_array_equal(fit.final_model.centers, centers)
    np.testing.assert_array_equal(fit.final_model.sigmas, sigmas)
    np.testing.assert_array_equal(fit.final_model.consequents, consequents)


def test_fit_rejects_nonpositive_radius_and_alpha(monkeypatch):
    _fake_optimizer(monkeypatch)
    raw = _raw_series()
    with pytest.raises(ValueError, match="radius"):
        fit_pc_nfpso_v2(
            raw,
            n_lags=5,
            validation_size=8,
            radius=0.0,
            alpha=1e-4,
            optimizer_seed=31001,
        )
    with pytest.raises(ValueError, match="alpha"):
        fit_pc_nfpso_v2(
            raw,
            n_lags=5,
            validation_size=8,
            radius=0.55,
            alpha=0.0,
            optimizer_seed=31001,
        )
