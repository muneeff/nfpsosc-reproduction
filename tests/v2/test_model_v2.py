import numpy as np
import pytest
from scipy.special import logsumexp

from nfpsosc.v2.model import TSKModel


def _model_two_rules() -> TSKModel:
    return TSKModel(
        centers=np.array([[0.0, 0.0], [1.0, -1.0]]),
        sigmas=np.array([[1.0, 2.0], [0.5, 1.5]]),
        consequents=np.array(
            [
                [1.2, -0.7, 0.3],
                [-0.4, 0.9, -0.2],
            ]
        ),
    )


def _finite_difference_jacobian(
    model: TSKModel,
    X: np.ndarray,
    step: float = 1e-6,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    out = np.empty_like(X)
    for i in range(len(X)):
        for j in range(X.shape[1]):
            plus = X[i].copy()
            minus = X[i].copy()
            plus[j] += step
            minus[j] -= step
            out[i, j] = (
                model.predict(plus)[0] - model.predict(minus)[0]
            ) / (2.0 * step)
    return out


def test_constructor_rejects_nonpositive_widths():
    with pytest.raises(ValueError, match="strictly positive"):
        TSKModel(
            centers=np.array([[0.0]]),
            sigmas=np.array([[0.0]]),
            consequents=np.array([[1.0, 0.0]]),
        )

    with pytest.raises(ValueError, match="strictly positive"):
        TSKModel(
            centers=np.array([[0.0]]),
            sigmas=np.array([[-1.0]]),
            consequents=np.array([[1.0, 0.0]]),
        )


def test_log_firing_strength_matches_gaussian_definition():
    model = TSKModel(
        centers=np.array([[1.0, 2.0]]),
        sigmas=np.array([[2.0, 4.0]]),
        consequents=np.zeros((1, 3)),
    )
    X = np.array([[3.0, 6.0]])
    expected = -0.5 * ((2.0 / 2.0) ** 2 + (4.0 / 4.0) ** 2)
    assert model.log_firing_strengths(X)[0, 0] == pytest.approx(expected)


def test_normalized_weights_sum_to_one_in_extreme_low_coverage():
    model = TSKModel(
        centers=np.array([[0.0], [0.5]]),
        sigmas=np.array([[1.0], [1.0]]),
        consequents=np.array([[1.0, 0.0], [2.0, 1.0]]),
    )
    X = np.array([[40.0], [50.0]])
    raw = model.firing_strengths(X)
    weights = model.normalized_weights(X)

    # At least one row is sufficiently far away that direct raw firing underflows.
    assert np.any(raw == 0.0)
    np.testing.assert_allclose(
        np.sum(weights, axis=1),
        np.ones(len(X)),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.all(np.isfinite(weights))


def test_log_total_firing_is_stable_and_matches_logsumexp():
    model = _model_two_rules()
    X = np.array([[0.2, -0.3], [8.0, 7.0]])
    log_raw = model.log_firing_strengths(X)
    np.testing.assert_allclose(
        model.log_total_firing_strength(X),
        logsumexp(log_raw, axis=1),
        rtol=0.0,
        atol=1e-14,
    )


def test_single_rule_prediction_is_exact_affine_consequent():
    model = TSKModel(
        centers=np.array([[10.0, -10.0]]),
        sigmas=np.array([[0.3, 0.8]]),
        consequents=np.array([[2.0, -3.0, 4.0]]),
    )
    X = np.array([[1.0, 2.0], [-2.0, 0.5]])
    expected = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 4.0
    np.testing.assert_allclose(model.predict(X), expected, atol=1e-14)


def test_consequent_design_matrix_is_rule_major_and_weighted():
    model = TSKModel(
        centers=np.array([[0.0], [1.0]]),
        sigmas=np.array([[1.0], [1.0]]),
        consequents=np.zeros((2, 2)),
    )
    X = np.array([[0.25], [0.75]])
    w = model.normalized_weights(X)
    design = model.consequent_design_matrix(X)

    expected = np.column_stack(
        [
            w[:, 0] * X[:, 0],
            w[:, 0],
            w[:, 1] * X[:, 0],
            w[:, 1],
        ]
    )
    np.testing.assert_allclose(design, expected, atol=1e-15)


def test_ridge_fit_matches_explicit_linear_system_and_regularizes_intercepts():
    model = TSKModel(
        centers=np.array([[0.0], [1.0]]),
        sigmas=np.array([[0.8], [0.8]]),
        consequents=np.zeros((2, 2)),
    )
    X = np.array([[-1.0], [0.0], [0.5], [1.0], [2.0]])
    y = np.array([-0.8, 0.2, 1.0, 1.7, 2.8])
    alpha = 0.05

    design = model.consequent_design_matrix(X)
    expected = np.linalg.solve(
        design.T @ design + alpha * np.eye(design.shape[1]),
        design.T @ y,
    ).reshape(2, 2)

    fitted = model.copy().fit_consequents(X, y, alpha=alpha)
    np.testing.assert_allclose(fitted.consequents, expected, rtol=1e-12, atol=1e-12)


def test_ridge_alpha_is_mandatory_and_positive():
    model = _model_two_rules()
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0])

    with pytest.raises(TypeError):
        model.fit_consequents(X, y)

    with pytest.raises(ValueError, match="positive"):
        model.fit_consequents(X, y, alpha=0.0)


def test_antecedent_vector_roundtrip():
    model = _model_two_rules()
    vector = model.antecedent_vector()
    rebuilt = model.with_antecedents(vector)

    np.testing.assert_array_equal(rebuilt.centers, model.centers)
    np.testing.assert_array_equal(rebuilt.sigmas, model.sigmas)
    np.testing.assert_array_equal(rebuilt.consequents, model.consequents)


def test_with_antecedents_rejects_nonpositive_width():
    model = _model_two_rules()
    vector = model.antecedent_vector()
    n = model.n_rules * model.n_features
    vector[n] = 0.0

    with pytest.raises(ValueError, match="strictly positive"):
        model.with_antecedents(vector)


def test_analytic_jacobian_matches_finite_difference():
    model = _model_two_rules()
    X = np.array(
        [
            [0.2, -0.1],
            [0.8, -0.6],
            [-0.5, 0.7],
        ]
    )
    analytic = model.jacobian(X)
    numeric = _finite_difference_jacobian(model, X)

    np.testing.assert_allclose(
        analytic,
        numeric,
        rtol=2e-6,
        atol=2e-7,
    )


def test_analytic_jacobian_matches_finite_difference_in_low_coverage_region():
    model = TSKModel(
        centers=np.array([[0.0], [0.5]]),
        sigmas=np.array([[1.0], [1.0]]),
        consequents=np.array([[0.7, -0.2], [-1.1, 0.4]]),
    )
    X = np.array([[10.0], [12.0]])
    assert np.all(model.log_total_firing_strength(X) < np.log(1e-4))

    analytic = model.jacobian(X)
    numeric = _finite_difference_jacobian(model, X, step=1e-5)

    np.testing.assert_allclose(
        analytic,
        numeric,
        rtol=2e-5,
        atol=2e-6,
    )


def test_input_validation_rejects_nonfinite_features():
    model = _model_two_rules()
    with pytest.raises(ValueError, match="finite"):
        model.predict(np.array([[np.nan, 0.0]]))
