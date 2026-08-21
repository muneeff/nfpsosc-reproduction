from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .initialization import (
    FrozenMinMaxScaler1D,
    TSKInitialization,
    initialize_tsk_from_subtractive_clustering,
)
from .model import TSKModel
from .objective import (
    CandidateObjectiveResult,
    ObjectiveWeights,
    evaluate_antecedent_candidate,
)
from .optimizer import PSOConfig, PSOResult, particle_swarm_optimize


FROZEN_PARTICLES = 12
FROZEN_ITERATIONS = 40


def _finite_series(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if len(arr) < 1:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def make_chronological_lags(
    scaled_series: np.ndarray,
    *,
    n_lags: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build contiguous chronological raw-lag supervised samples."""
    y = _finite_series(scaled_series, name="scaled_series")
    if not isinstance(n_lags, (int, np.integer)) or int(n_lags) < 1:
        raise ValueError("n_lags must be a positive integer.")
    L = int(n_lags)
    if len(y) <= L:
        raise ValueError("Series is too short for the requested lag dimension.")
    X = np.asarray([y[t - L : t] for t in range(L, len(y))], dtype=float)
    target = y[L:].copy()
    return X, target


@dataclass(frozen=True)
class PCNFPSOTrainingData:
    """Leakage-isolated pre-test data used by V2 PC-NFPSO."""

    raw_pretest: np.ndarray
    scaler: FrozenMinMaxScaler1D
    n_lags: int
    validation_size: int
    fitting_raw_end: int
    X_fit: np.ndarray
    y_fit: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_pretest: np.ndarray
    y_pretest: np.ndarray


@dataclass(frozen=True)
class PCNFPSOFit:
    """Result of antecedent selection followed by full-pretest consequent refit."""

    training: PCNFPSOTrainingData
    initialization: TSKInitialization
    optimizer_result: PSOResult
    selected_candidate: CandidateObjectiveResult
    final_model: TSKModel
    radius: float
    alpha: float
    optimizer_seed: int
    dynamics: str
    boundary: str
    use_differencing: bool = False


def prepare_pc_nfpso_training_data(
    raw_pretest: np.ndarray,
    *,
    n_lags: int,
    validation_size: int,
) -> PCNFPSOTrainingData:
    """Prepare the frozen scaled fit/validation split without test access."""
    raw = _finite_series(raw_pretest, name="raw_pretest")
    if not isinstance(n_lags, (int, np.integer)) or int(n_lags) < 1:
        raise ValueError("n_lags must be a positive integer.")
    if not isinstance(validation_size, (int, np.integer)) or int(validation_size) < 1:
        raise ValueError("validation_size must be a positive integer.")

    L = int(n_lags)
    V = int(validation_size)
    n_supervised = len(raw) - L
    if n_supervised <= V:
        raise ValueError(
            "Pre-test series must leave at least one fitting sample before validation."
        )
    n_fit = n_supervised - V
    fitting_raw_end = L + n_fit
    if fitting_raw_end < 2:
        raise ValueError("Fitting scaler requires at least two raw observations.")

    scaler = FrozenMinMaxScaler1D.fit(raw[:fitting_raw_end])
    scaled = scaler.transform(raw)
    X_all, y_all = make_chronological_lags(scaled, n_lags=L)
    X_fit = X_all[:n_fit].copy()
    y_fit = y_all[:n_fit].copy()
    X_validation = X_all[n_fit:].copy()
    y_validation = y_all[n_fit:].copy()

    if len(X_validation) != V:
        raise RuntimeError("Internal validation split length mismatch.")

    return PCNFPSOTrainingData(
        raw_pretest=raw.copy(),
        scaler=scaler,
        n_lags=L,
        validation_size=V,
        fitting_raw_end=fitting_raw_end,
        X_fit=X_fit,
        y_fit=y_fit,
        X_validation=X_validation,
        y_validation=y_validation,
        X_pretest=X_all.copy(),
        y_pretest=y_all.copy(),
    )


def fit_pc_nfpso_v2(
    raw_pretest: np.ndarray,
    *,
    n_lags: int,
    validation_size: int,
    radius: float,
    alpha: float,
    optimizer_seed: int,
    dynamics: Literal["nonconstricted", "constricted"] = "constricted",
    boundary: Literal["project", "feasible_rejection"] = "project",
    objective_weights: ObjectiveWeights = ObjectiveWeights(),
    use_differencing: bool = False,
) -> PCNFPSOFit:
    """Fit V2 PC-NFPSO with optional differencing for non-stationary trends."""
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be a finite positive scalar.")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be a finite positive scalar.")
    if not isinstance(optimizer_seed, (int, np.integer)):
        raise ValueError("optimizer_seed must be an integer.")

    raw = _finite_series(raw_pretest, name="raw_pretest")
    
    if use_differencing and len(raw) > 1:
        data_to_fit = np.diff(raw)
    else:
        data_to_fit = raw

    training = prepare_pc_nfpso_training_data(
        data_to_fit,
        n_lags=n_lags,
        validation_size=validation_size,
    )
    initialization = initialize_tsk_from_subtractive_clustering(
        training.X_fit,
        training.y_fit,
        radius=float(radius),
    )

    def cost(antecedents: np.ndarray) -> float:
        result = evaluate_antecedent_candidate(
            initialization.model,
            antecedents,
            training.X_fit,
            training.y_fit,
            training.X_validation,
            training.y_validation,
            alpha=float(alpha),
            weights=objective_weights,
        )
        return float(result.cost)

    optimizer_config = PSOConfig(
        iterations=FROZEN_ITERATIONS,
        particles=FROZEN_PARTICLES,
        seed=int(optimizer_seed),
        dynamics=dynamics,
        boundary=boundary,
    )
    optimizer_result = particle_swarm_optimize(
        cost_function=cost,
        initial_position=initialization.particle0,
        lower_bound=initialization.bounds.lower,
        upper_bound=initialization.bounds.upper,
        config=optimizer_config,
    )

    best_position = np.asarray(optimizer_result.best_position, dtype=float).reshape(-1)
    
    selected_candidate = evaluate_antecedent_candidate(
        initialization.model,
        best_position,
        training.X_fit,
        training.y_fit,
        training.X_validation,
        training.y_validation,
        alpha=float(alpha),
        weights=objective_weights,
    )

    final_model = initialization.model.with_antecedents(best_position)
    final_model.fit_consequents(
        training.X_pretest,
        training.y_pretest,
        alpha=float(alpha),
    )

    return PCNFPSOFit(
        training=training,
        initialization=initialization,
        optimizer_result=optimizer_result,
        selected_candidate=selected_candidate,
        final_model=final_model,
        radius=float(radius),
        alpha=float(alpha),
        optimizer_seed=int(optimizer_seed),
        dynamics=str(dynamics),
        boundary=str(boundary),
        use_differencing=use_differencing,
    )


def forecast_pc_nfpso_v2(
    fitted: PCNFPSOFit,
    test_actuals: np.ndarray,
) -> np.ndarray:
    """One-step test forecasts with fixed parameters and observed true history."""
    actuals = _finite_series(test_actuals, name="test_actuals")
    raw_history = fitted.training.raw_pretest.astype(float, copy=True)
    L = int(fitted.training.n_lags)

    predictions = np.empty(len(actuals), dtype=float)

    if getattr(fitted, "use_differencing", False):
        curr_last_val = raw_history[-1]
        diff_history = np.diff(raw_history)
        
        for i, observed in enumerate(actuals):
            lag_raw_diff = diff_history[-L:]
            lag_scaled = fitted.training.scaler.transform(lag_raw_diff).reshape(1, -1)
            pred_diff_scaled = float(fitted.final_model.predict(lag_scaled)[0])
            pred_diff_raw = float(
                fitted.training.scaler.inverse_transform(np.asarray([pred_diff_scaled]))[0]
            )
            predictions[i] = curr_last_val + pred_diff_raw
            
            new_diff = observed - curr_last_val
            diff_history = np.append(diff_history, float(new_diff))
            curr_last_val = float(observed)
    else:
        for i, observed in enumerate(actuals):
            lag_raw = raw_history[-L:]
            lag_scaled = fitted.training.scaler.transform(lag_raw).reshape(1, -1)
            pred_scaled = float(fitted.final_model.predict(lag_scaled)[0])
            pred_raw = float(
                fitted.training.scaler.inverse_transform(np.asarray([pred_scaled]))[0]
            )
            predictions[i] = pred_raw
            raw_history = np.append(raw_history, float(observed))

    return predictions