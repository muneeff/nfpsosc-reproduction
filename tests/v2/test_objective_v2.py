import numpy as np
import pytest

from nfpsosc.v2.model import TSKModel
from nfpsosc.v2.objective import (
    ACTIVATION_FLOOR,
    ACTIVATION_NUMERICAL_FLOOR,
    CandidateObjectiveResult,
    ObjectiveWeights,
    composite_objective_components,
    evaluate_antecedent_candidate,
    low_coverage_penalty,
    rmse,
    sensitivity_penalty,
)


def _template() -> TSKModel:
    return TSKModel(
        centers=np.array([[0.25, 0.25], [0.75, 0.75]], dtype=float),
        sigmas=np.array([[0.30, 0.35], [0.25, 0.30]], dtype=float),
        consequents=np.zeros((2, 3), dtype=float),
    )


def _data():
    X_fit = np.array(
        [[0.10, 0.20], [0.20, 0.35], [0.35, 0.40], [0.55, 0.60], [0.70, 0.80], [0.85, 0.75]],
        dtype=float,
    )
    y_fit = 0.2 + 0.7 * X_fit[:, 0] - 0.15 * X_fit[:, 1]
    X_val = np.array([[0.30, 0.50], [0.60, 0.70], [0.90, 0.85]], dtype=float)
    y_val = 0.2 + 0.7 * X_val[:, 0] - 0.15 * X_val[:, 1]
    return X_fit, y_fit, X_val, y_val


def test_rmse_matches_manual_value():
    y = np.array([1.0, 2.0, 4.0])
    p = np.array([0.0, 2.0, 5.0])
    assert rmse(y, p) == pytest.approx(np.sqrt(2.0 / 3.0))


def test_sensitivity_penalty_matches_manual_jacobian_norm():
    model = TSKModel(
        centers=np.array([[0.5, 0.5]]),
        sigmas=np.array([[0.2, 0.2]]),
        consequents=np.array([[2.0, -3.0, 1.0]]),
    )
    Xf = np.array([[0.2, 0.4], [0.4, 0.6]])
    Xv = np.array([[0.7, 0.8]])
    # One rule => normalized weight is identically 1, so df/dx = [2,-3].
    assert sensitivity_penalty(model, Xf, Xv) == pytest.approx(13.0)


def test_low_coverage_zero_when_total_firing_exceeds_floor():
    model = TSKModel(
        centers=np.array([[0.5]]),
        sigmas=np.array([[1.0]]),
        consequents=np.array([[0.0, 0.0]]),
    )
    Xv = np.array([[0.5], [0.6]])
    assert low_coverage_penalty(model, Xv) == pytest.approx(0.0)


def test_low_coverage_matches_frozen_log_shortfall_with_numerical_floor():
    model = TSKModel(
        centers=np.array([[0.0]]),
        sigmas=np.array([[0.01]]),
        consequents=np.array([[0.0, 0.0]]),
    )
    Xv = np.array([[10.0]])
    expected = (
        np.log(ACTIVATION_FLOOR) - np.log(ACTIVATION_NUMERICAL_FLOOR)
    ) ** 2
    assert low_coverage_penalty(model, Xv) == pytest.approx(expected)


def test_composite_total_matches_exact_frozen_formula():
    Xf, yf, Xv, yv = _data()
    model = _template().fit_consequents(Xf, yf, alpha=1e-3)
    c = composite_objective_components(model, Xf, yf, Xv, yv)
    expected = (
        0.5 * c.fit_rmse
        + c.validation_rmse
        + 0.001 * c.sensitivity_penalty
        + 0.01 * c.low_coverage_penalty
    )
    assert c.total == pytest.approx(expected, rel=0.0, abs=1e-15)


def test_candidate_returns_fitted_copy_and_does_not_mutate_template():
    Xf, yf, Xv, yv = _data()
    template = _template()
    before = template.consequents.copy()
    result = evaluate_antecedent_candidate(
        template,
        template.antecedent_vector(),
        Xf,
        yf,
        Xv,
        yv,
        alpha=1e-3,
    )
    assert isinstance(result, CandidateObjectiveResult)
    assert np.array_equal(template.consequents, before)
    assert not np.array_equal(result.model.consequents, before)
    assert np.isfinite(result.cost)


def test_validation_targets_change_cost_but_not_fitted_consequents():
    Xf, yf, Xv, yv = _data()
    template = _template()
    antecedents = template.antecedent_vector()

    r1 = evaluate_antecedent_candidate(
        template, antecedents, Xf, yf, Xv, yv, alpha=1e-3
    )
    altered_yv = yv + np.array([5.0, -4.0, 3.0])
    r2 = evaluate_antecedent_candidate(
        template, antecedents, Xf, yf, Xv, altered_yv, alpha=1e-3
    )

    np.testing.assert_array_equal(r1.model.consequents, r2.model.consequents)
    assert r1.components.fit_rmse == r2.components.fit_rmse
    assert r1.components.sensitivity_penalty == r2.components.sensitivity_penalty
    assert r1.components.low_coverage_penalty == r2.components.low_coverage_penalty
    assert r1.components.validation_rmse != r2.components.validation_rmse
    assert r1.cost != r2.cost


def test_validation_features_affect_objective_but_are_not_used_to_fit_consequents():
    Xf, yf, Xv, yv = _data()
    template = _template()
    antecedents = template.antecedent_vector()
    r1 = evaluate_antecedent_candidate(
        template, antecedents, Xf, yf, Xv, yv, alpha=1e-3
    )
    Xv2 = Xv + 0.05
    r2 = evaluate_antecedent_candidate(
        template, antecedents, Xf, yf, Xv2, yv, alpha=1e-3
    )
    np.testing.assert_array_equal(r1.model.consequents, r2.model.consequents)
    assert r1.cost != r2.cost


def test_objective_uses_fit_plus_validation_for_sensitivity():
    Xf, yf, Xv, yv = _data()
    model = _template().fit_consequents(Xf, yf, alpha=1e-3)
    c = composite_objective_components(model, Xf, yf, Xv, yv)
    jac = model.jacobian(np.vstack([Xf, Xv]))
    expected = np.mean(np.sum(jac * jac, axis=1))
    assert c.sensitivity_penalty == pytest.approx(expected)


def test_objective_ablation_weights_are_supported_explicitly():
    Xf, yf, Xv, yv = _data()
    model = _template().fit_consequents(Xf, yf, alpha=1e-3)
    validation_only = ObjectiveWeights(
        fit_rmse=0.0,
        validation_rmse=1.0,
        sensitivity=0.0,
        low_coverage=0.0,
    )
    c = composite_objective_components(
        model, Xf, yf, Xv, yv, weights=validation_only
    )
    assert c.total == pytest.approx(c.validation_rmse)


def test_candidate_requires_explicit_alpha():
    Xf, yf, Xv, yv = _data()
    template = _template()
    with pytest.raises(TypeError):
        evaluate_antecedent_candidate(  # type: ignore[call-arg]
            template,
            template.antecedent_vector(),
            Xf,
            yf,
            Xv,
            yv,
        )


def test_rejects_nonfinite_and_empty_validation_data():
    Xf, yf, Xv, yv = _data()
    model = _template().fit_consequents(Xf, yf, alpha=1e-3)
    with pytest.raises(ValueError):
        composite_objective_components(model, Xf, yf, Xv[:0], yv[:0])
    bad = Xv.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        composite_objective_components(model, Xf, yf, bad, yv)


def test_objective_weights_reject_negative_or_nonfinite_values():
    with pytest.raises(ValueError):
        ObjectiveWeights(sensitivity=-1.0)
    with pytest.raises(ValueError):
        ObjectiveWeights(low_coverage=np.inf)
