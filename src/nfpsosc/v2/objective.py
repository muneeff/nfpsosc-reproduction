from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import TSKModel


FIT_RMSE_WEIGHT = 0.5
VALIDATION_RMSE_WEIGHT = 1.0
SENSITIVITY_WEIGHT = 0.001
LOW_COVERAGE_WEIGHT = 0.01
ACTIVATION_FLOOR = 1e-4
ACTIVATION_NUMERICAL_FLOOR = 1e-300


@dataclass(frozen=True)
class ObjectiveWeights:
    """Frozen V2 composite-objective weights.

    Alternative values are permitted only for the pre-specified objective
    ablations. The base PC-NFPSO uses the defaults below exactly.
    """

    fit_rmse: float = FIT_RMSE_WEIGHT
    validation_rmse: float = VALIDATION_RMSE_WEIGHT
    sensitivity: float = SENSITIVITY_WEIGHT
    low_coverage: float = LOW_COVERAGE_WEIGHT

    def __post_init__(self) -> None:
        values = (
            self.fit_rmse,
            self.validation_rmse,
            self.sensitivity,
            self.low_coverage,
        )
        if not all(np.isfinite(v) and v >= 0.0 for v in values):
            raise ValueError("Objective weights must be finite and non-negative.")


@dataclass(frozen=True)
class ObjectiveComponents:
    fit_rmse: float
    validation_rmse: float
    sensitivity_penalty: float
    low_coverage_penalty: float
    total: float


@dataclass(frozen=True)
class CandidateObjectiveResult:
    """Detailed result for one antecedent candidate.

    ``model`` contains consequents fitted exclusively on the fitting segment.
    """

    model: TSKModel
    components: ObjectiveComponents

    @property
    def cost(self) -> float:
        return self.components.total


def _as_xy(X: np.ndarray, y: np.ndarray, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    X2 = np.asarray(X, dtype=float)
    y1 = np.asarray(y, dtype=float).reshape(-1)
    if X2.ndim != 2 or len(X2) < 1:
        raise ValueError(f"{name} X must be a non-empty 2D array.")
    if len(X2) != len(y1):
        raise ValueError(f"{name} X and y must contain the same number of samples.")
    if not np.all(np.isfinite(X2)) or not np.all(np.isfinite(y1)):
        raise ValueError(f"{name} data must contain only finite values.")
    return X2, y1


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(truth) < 1 or len(truth) != len(pred):
        raise ValueError("RMSE inputs must be non-empty and have equal length.")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(pred)):
        raise ValueError("RMSE inputs must be finite.")
    value = float(np.sqrt(np.mean((truth - pred) ** 2)))
    if not np.isfinite(value):
        raise ValueError("RMSE is non-finite.")
    return value


def sensitivity_penalty(model: TSKModel, X_fit: np.ndarray, X_validation: np.ndarray) -> float:
    Xf = np.asarray(X_fit, dtype=float)
    Xv = np.asarray(X_validation, dtype=float)
    if Xf.ndim != 2 or Xv.ndim != 2 or len(Xf) < 1 or len(Xv) < 1:
        raise ValueError("Sensitivity requires non-empty 2D fitting and validation matrices.")
    if Xf.shape[1] != Xv.shape[1]:
        raise ValueError("Fitting and validation feature dimensions must match.")
    X_union = np.vstack([Xf, Xv])
    jac = model.jacobian(X_union)
    value = float(np.mean(np.sum(jac * jac, axis=1)))
    if not np.isfinite(value):
        raise ValueError("Sensitivity penalty is non-finite.")
    return value


def low_coverage_penalty(
    model: TSKModel,
    X_validation: np.ndarray,
    *,
    activation_floor: float = ACTIVATION_FLOOR,
    numerical_floor: float = ACTIVATION_NUMERICAL_FLOOR,
) -> float:
    Xv = np.asarray(X_validation, dtype=float)
    if Xv.ndim != 2 or len(Xv) < 1:
        raise ValueError("Coverage penalty requires a non-empty 2D validation matrix.")
    if not np.isfinite(activation_floor) or not 0.0 < activation_floor:
        raise ValueError("activation_floor must be finite and positive.")
    if not np.isfinite(numerical_floor) or not 0.0 < numerical_floor <= activation_floor:
        raise ValueError(
            "numerical_floor must be finite, positive, and no larger than activation_floor."
        )

    log_total = model.log_total_firing_strength(Xv)
    log_floor = float(np.log(activation_floor))
    log_numeric = float(np.log(numerical_floor))
    effective_log_total = np.maximum(log_total, log_numeric)
    shortfall = np.maximum(0.0, log_floor - effective_log_total)
    value = float(np.mean(shortfall * shortfall))
    if not np.isfinite(value):
        raise ValueError("Low-coverage penalty is non-finite.")
    return value


def composite_objective_components(
    model: TSKModel,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    weights: ObjectiveWeights = ObjectiveWeights(),
    activation_floor: float = ACTIVATION_FLOOR,
    numerical_floor: float = ACTIVATION_NUMERICAL_FLOOR,
) -> ObjectiveComponents:
    """Evaluate the frozen V2 objective for an already fitted TSK model."""
    Xf, yf = _as_xy(X_fit, y_fit, name="fitting")
    Xv, yv = _as_xy(X_validation, y_validation, name="validation")
    if Xf.shape[1] != model.n_features or Xv.shape[1] != model.n_features:
        raise ValueError("Objective feature dimension does not match the TSK model.")

    fit_error = rmse(yf, model.predict(Xf))
    validation_error = rmse(yv, model.predict(Xv))
    sensitivity = sensitivity_penalty(model, Xf, Xv)
    coverage = low_coverage_penalty(
        model,
        Xv,
        activation_floor=activation_floor,
        numerical_floor=numerical_floor,
    )
    total = float(
        weights.fit_rmse * fit_error
        + weights.validation_rmse * validation_error
        + weights.sensitivity * sensitivity
        + weights.low_coverage * coverage
    )
    if not np.isfinite(total):
        raise ValueError("Composite objective is non-finite.")
    return ObjectiveComponents(
        fit_rmse=fit_error,
        validation_rmse=validation_error,
        sensitivity_penalty=sensitivity,
        low_coverage_penalty=coverage,
        total=total,
    )


def evaluate_antecedent_candidate(
    template: TSKModel,
    antecedent_vector: np.ndarray,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    alpha: float,
    weights: ObjectiveWeights = ObjectiveWeights(),
    activation_floor: float = ACTIVATION_FLOOR,
    numerical_floor: float = ACTIVATION_NUMERICAL_FLOOR,
) -> CandidateObjectiveResult:
    """Fit candidate consequents on fitting data only, then evaluate V2 objective.

    Validation targets are never passed to the consequent ridge solve. This is
    the variable-projection objective used by the V2 antecedent optimizer.
    """
    Xf, yf = _as_xy(X_fit, y_fit, name="fitting")
    Xv, yv = _as_xy(X_validation, y_validation, name="validation")
    if Xf.shape[1] != template.n_features or Xv.shape[1] != template.n_features:
        raise ValueError("Candidate feature dimension does not match the TSK template.")

    candidate = template.with_antecedents(antecedent_vector)
    candidate.fit_consequents(Xf, yf, alpha=alpha)
    components = composite_objective_components(
        candidate,
        Xf,
        yf,
        Xv,
        yv,
        weights=weights,
        activation_floor=activation_floor,
        numerical_floor=numerical_floor,
    )
    return CandidateObjectiveResult(model=candidate, components=components)
