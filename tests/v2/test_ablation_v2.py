from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import nfpsosc.v2.ablation as ab
import nfpsosc.v2.pc_nfpso as pc
from nfpsosc.v2.initialization import initialize_tsk_from_subtractive_clustering
from nfpsosc.v2.objective import ObjectiveWeights
from nfpsosc.v2.pc_nfpso import prepare_pc_nfpso_training_data

def _raw_series(n: int = 54) -> np.ndarray:
    t = np.arange(n, dtype=float)
    return (
        20.0
        + 0.035 * t
        + 1.1 * np.sin(2.0 * np.pi * t / 12.0)
        + 0.15 * np.cos(0.37 * t)
    )

def _locked_config():
    return json.loads(
        Path("configs/v2/ablation_v2.json").read_text(encoding="utf-8")
    )

def test_variant_order_matches_locked_contract_exactly():
    cfg = _locked_config()
    assert tuple(cfg["synthetic_final_ablation"]["variants"]) == ab.VARIANT_ORDER
    assert tuple(ab.VARIANT_SPECS) == ab.VARIANT_ORDER

def test_variant_factor_mapping_matches_locked_contract():
    cfg = _locked_config()["variants"]
    for name in ab.VARIANT_ORDER:
        spec = ab.VARIANT_SPECS[name]
        frozen = cfg[name]
        assert spec.pso is bool(frozen["pso"])
        if name == "NF_BASE":
            assert spec.dynamics is None
            assert spec.boundary is None
            assert spec.objective_weights is None
        else:
            assert spec.dynamics == frozen["dynamics"]
            assert spec.boundary == frozen["boundary"]

def test_objective_ablation_weights_are_exact():
    assert ab.FULL_OBJECTIVE == ObjectiveWeights(
        fit_rmse=0.5,
        validation_rmse=1.0,
        sensitivity=0.001,
        low_coverage=0.01,
    )
    assert ab.NO_SENSITIVITY_OBJECTIVE == ObjectiveWeights(
        fit_rmse=0.5,
        validation_rmse=1.0,
        sensitivity=0.0,
        low_coverage=0.01,
    )
    assert ab.NO_COVERAGE_OBJECTIVE == ObjectiveWeights(
        fit_rmse=0.5,
        validation_rmse=1.0,
        sensitivity=0.001,
        low_coverage=0.0,
    )
    assert ab.VALIDATION_ONLY_OBJECTIVE == ObjectiveWeights(
        fit_rmse=0.0,
        validation_rmse=1.0,
        sensitivity=0.0,
        low_coverage=0.0,
    )

def test_pc_fit_annotation_matches_optimizer_runtime_vocabulary():
    ann = inspect.signature(pc.fit_pc_nfpso_v2).parameters["dynamics"].annotation
    assert "nonconstricted" in str(ann)
    assert "standard" not in str(ann)

@pytest.mark.parametrize(
    "variant,expected_dynamics,expected_boundary,expected_weights",
    [
        ("NFPSO", "nonconstricted", "feasible_rejection", ab.FULL_OBJECTIVE),
        ("P_NFPSO", "nonconstricted", "project", ab.FULL_OBJECTIVE),
        ("C_NFPSO", "constricted", "feasible_rejection", ab.FULL_OBJECTIVE),
        ("PC_NFPSO", "constricted", "project", ab.FULL_OBJECTIVE),
        ("PC_NO_SENSITIVITY", "constricted", "project", ab.NO_SENSITIVITY_OBJECTIVE),
        ("PC_NO_COVERAGE", "constricted", "project", ab.NO_COVERAGE_OBJECTIVE),
        ("PC_VALIDATION_ONLY", "constricted", "project", ab.VALIDATION_ONLY_OBJECTIVE),
    ],
)
def test_stochastic_dispatch_changes_only_frozen_variant_factors(
    monkeypatch,
    variant,
    expected_dynamics,
    expected_boundary,
    expected_weights,
):
    calls = []
    def fake_fit(raw_pretest, **kwargs):
        calls.append((np.asarray(raw_pretest, dtype=float).copy(), dict(kwargs)))
        return SimpleNamespace(training="T", final_model="M")

    monkeypatch.setattr(ab, "fit_pc_nfpso_v2", fake_fit)

    raw = _raw_series()
    result = ab.fit_ablation_variant_v2(
        variant,
        raw,
        n_lags=5,
        validation_size=8,
        radius=1.0,
        alpha=0.05,
        optimizer_seed=41001,
    )
    assert result is not None
    assert len(calls) == 1
    observed_raw, kwargs = calls[0]
    np.testing.assert_array_equal(observed_raw, raw)
    assert kwargs == {
        "n_lags": 5,
        "validation_size": 8,
        "radius": 1.0,
        "alpha": 0.05,
        "optimizer_seed": 41001,
        "dynamics": expected_dynamics,
        "boundary": expected_boundary,
        "objective_weights": expected_weights,
    }

def test_nf_base_is_clustering_only_then_full_pretest_consequent_refit(monkeypatch):
    def forbidden_pso(*args, **kwargs):
        raise AssertionError("NF_BASE must not invoke PC-NFPSO/PSO.")
    monkeypatch.setattr(ab, "fit_pc_nfpso_v2", forbidden_pso)

    raw = _raw_series()
    fit = ab.fit_ablation_variant_v2(
        "NF_BASE",
        raw,
        n_lags=5,
        validation_size=8,
        radius=1.0,
        alpha=0.05,
        optimizer_seed=None,
    )

    training = prepare_pc_nfpso_training_data(
        raw,
        n_lags=5,
        validation_size=8,
    )
    initialization = initialize_tsk_from_subtractive_clustering(
        training.X_fit,
        training.y_fit,
        radius=1.0,
    )
    expected = initialization.model.copy()
    expected.fit_consequents(
        training.X_pretest,
        training.y_pretest,
        alpha=0.05,
    )

    np.testing.assert_allclose(fit.final_model.centers, expected.centers, rtol=0, atol=0)
    np.testing.assert_allclose(fit.final_model.sigmas, expected.sigmas, rtol=0, atol=0)
    np.testing.assert_allclose(
        fit.final_model.consequents,
        expected.consequents,
        rtol=1e-13,
        atol=1e-13,
    )
    assert fit.optimizer_seed is None
    assert fit.dynamics is None
    assert fit.boundary is None
    assert fit.variant == "NF_BASE"

def test_selected_radius_one_sc_initialization_requires_no_clipping():
    raw = _raw_series()
    training = prepare_pc_nfpso_training_data(
        raw,
        n_lags=5,
        validation_size=8,
    )
    initialization = initialize_tsk_from_subtractive_clustering(
        training.X_fit,
        training.y_fit,
        radius=1.0,
    )
    np.testing.assert_allclose(
        initialization.particle0,
        initialization.particle0_unclipped,
        rtol=0,
        atol=0,
    )

def test_nf_base_rejects_optimizer_seed():
    with pytest.raises(ValueError, match="deterministic"):
        ab.fit_ablation_variant_v2(
            "NF_BASE",
            _raw_series(),
            n_lags=5,
            validation_size=8,
            radius=1.0,
            alpha=0.05,
            optimizer_seed=41001,
        )

def test_stochastic_variant_requires_optimizer_seed():
    with pytest.raises(ValueError, match="requires"):
        ab.fit_ablation_variant_v2(
            "NFPSO",
            _raw_series(),
            n_lags=5,
            validation_size=8,
            radius=1.0,
            alpha=0.05,
            optimizer_seed=None,
        )

def test_unknown_variant_fails_closed():
    with pytest.raises(ValueError, match="Unknown frozen"):
        ab.variant_spec("NOT_A_VARIANT")

def test_nf_base_forecast_uses_shared_frozen_forecast_path(monkeypatch):
    fit = ab.fit_nf_base_v2(
        _raw_series(),
        n_lags=5,
        validation_size=8,
        radius=1.0,
        alpha=0.05,
    )
    actual = np.array([22.0, 23.0, 24.0])
    expected = np.array([1.0, 2.0, 3.0])
    calls = []

    def fake_forecast(fitted, test_actuals):
        calls.append((fitted, np.asarray(test_actuals).copy()))
        return expected.copy()

    monkeypatch.setattr(ab, "forecast_pc_nfpso_v2", fake_forecast)
    observed = ab.forecast_ablation_v2(fit, actual)

    np.testing.assert_array_equal(observed, expected)
    assert calls[0][0] is fit
    np.testing.assert_array_equal(calls[0][1], actual)

def test_forecast_dispatch_reuses_exact_original_function(monkeypatch):
    sentinel_fit = SimpleNamespace()
    actual = np.array([1.0, 2.0])
    expected = np.array([3.0, 4.0])
    calls = []

    def fake_forecast(fitted, test_actuals):
        calls.append((fitted, np.asarray(test_actuals).copy()))
        return expected.copy()

    monkeypatch.setattr(ab, "forecast_pc_nfpso_v2", fake_forecast)
    observed = ab.forecast_ablation_v2(sentinel_fit, actual)

    np.testing.assert_array_equal(observed, expected)
    assert calls[0][0] is sentinel_fit
    np.testing.assert_array_equal(calls[0][1], actual)

def test_pc_nfpso_ablation_dispatch_is_numerically_identical_to_direct_path():
    """Dispatcher must preserve the full PC-NFPSO reference path exactly."""
    raw_full = _raw_series(60)
    raw_pretest = raw_full[:54]
    test_actuals = raw_full[54:]

    direct = pc.fit_pc_nfpso_v2(
        raw_pretest,
        n_lags=5,
        validation_size=8,
        radius=1.0,
        alpha=0.05,
        optimizer_seed=41001,
        dynamics="constricted",
        boundary="project",
        objective_weights=ab.FULL_OBJECTIVE,
    )

    via_ablation = ab.fit_ablation_variant_v2(
        "PC_NFPSO",
        raw_pretest,
        n_lags=5,
        validation_size=8,
        radius=1.0,
        alpha=0.05,
        optimizer_seed=41001,
    )

    np.testing.assert_array_equal(
        direct.optimizer_result.best_position,
        via_ablation.optimizer_result.best_position,
    )
    assert direct.optimizer_result.best_cost == via_ablation.optimizer_result.best_cost

    np.testing.assert_array_equal(
        direct.final_model.centers,
        via_ablation.final_model.centers,
    )
    np.testing.assert_array_equal(
        direct.final_model.sigmas,
        via_ablation.final_model.sigmas,
    )
    np.testing.assert_array_equal(
        direct.final_model.consequents,
        via_ablation.final_model.consequents,
    )

    pred_direct = pc.forecast_pc_nfpso_v2(direct, test_actuals)
    pred_ablation = ab.forecast_ablation_v2(via_ablation, test_actuals)
    np.testing.assert_array_equal(pred_direct, pred_ablation)
