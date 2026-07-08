from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .clustering import initialize_tsk_from_sc
from .data import (
    MinMaxScaler,
    chronological_split,
    create_lagged_data,
    load_revenue_series,
)
from .diagnostics import model_diagnostics
from .fis import TSKFIS
from .metrics import naive_scale, regression_metrics
from .optimizers import PSOConfig, particle_swarm_optimize
from .plots import save_prediction_plot, save_pso_history, save_residual_plot


@dataclass(frozen=True)
class ExperimentConfig:
    csv_path: str
    output_dir: str = "outputs/revenue_reproduction"
    n_lags: int = 5
    lag_order: str = "chronological"
    train_fraction: float = 0.85
    scale_min: float = 0.1
    scale_max: float = 0.9
    sc_radius: float = 0.55
    protocol: str = "legacy_intent"  # legacy_intent or research_baseline
    objective: str = "error_std"     # error_std, rmse, or composite
    sensitivity_weight: float = 0.0
    validation_fraction: float = 0.20
    activation_floor: float = 1e-4
    activation_penalty_weight: float = 0.01
    seed: int = 123
    iterations: int = 1000
    particles: int = 25


def _make_antecedent_bounds(
    model: TSKFIS, X_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bounds for centers and positive Gaussian widths only."""
    feature_min = np.min(X_train, axis=0)
    feature_max = np.max(X_train, axis=0)
    feature_range = np.maximum(feature_max - feature_min, 1e-3)
    center_lower = np.tile(feature_min - 0.05 * feature_range, model.n_rules)
    center_upper = np.tile(feature_max + 0.05 * feature_range, model.n_rules)
    sigma_lower = np.tile(0.05 * feature_range, model.n_rules)
    sigma_upper = np.tile(1.00 * feature_range, model.n_rules)
    return (
        np.concatenate([center_lower, sigma_lower]),
        np.concatenate([center_upper, sigma_upper]),
    )


def _antecedent_vector(model: TSKFIS) -> np.ndarray:
    return np.concatenate([model.centers.ravel(), model.sigmas.ravel()])


def _model_from_antecedents(template: TSKFIS, vector: np.ndarray) -> TSKFIS:
    v = np.asarray(vector, dtype=float).reshape(-1)
    n = template.n_rules * template.n_features
    if len(v) != 2 * n:
        raise ValueError(f"Expected {2*n} antecedent parameters, received {len(v)}.")
    model = template.copy()
    model.centers = v[:n].reshape(template.centers.shape)
    model.sigmas = v[n:].reshape(template.sigmas.shape)
    return model


def run_experiment(config: ExperimentConfig) -> dict:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_revenue_series(config.csv_path)
    raw = df["revenue_million_yer"].to_numpy(dtype=float)

    # Build raw supervised samples first so the split is determined strictly by time.
    X_raw, y_raw = create_lagged_data(raw, config.n_lags, config.lag_order)
    Xtr_raw, ytr_raw, Xte_raw, yte_raw = chronological_split(
        X_raw, y_raw, config.train_fraction
    )

    if config.protocol == "legacy_intent":
        # Matches the thesis table: scaler estimated from all 156 observations.
        scaler = MinMaxScaler.fit(raw, config.scale_min, config.scale_max)
    elif config.protocol == "research_baseline":
        # Leakage-free: scaler estimated only from observations available before test.
        raw_train_end = config.n_lags + len(ytr_raw)
        scaler = MinMaxScaler.fit(raw[:raw_train_end], config.scale_min, config.scale_max)
    else:
        raise ValueError(f"Unknown protocol: {config.protocol}")

    scaled = scaler.transform(raw)
    X, y = create_lagged_data(scaled, config.n_lags, config.lag_order)
    X_train, y_train, X_test, y_test = chronological_split(
        X, y, config.train_fraction
    )

    if config.protocol == "research_baseline":
        n_validation = max(8, int(np.floor(config.validation_fraction * len(X_train))))
        if n_validation >= len(X_train) - 10:
            raise ValueError("validation_fraction leaves too little fitting data.")
        X_fit, y_fit = X_train[:-n_validation], y_train[:-n_validation]
        X_validation, y_validation = X_train[-n_validation:], y_train[-n_validation:]
        initial_model, sc_result = initialize_tsk_from_sc(
            X_fit, y_fit, radius=config.sc_radius
        )
    else:
        X_fit, y_fit = X_train, y_train
        X_validation = np.empty((0, X_train.shape[1]))
        y_validation = np.empty((0,))
        initial_model, sc_result = initialize_tsk_from_sc(
            X_train, y_train, radius=config.sc_radius
        )
    p0 = initial_model.to_vector()

    def objective_for_model(model: TSKFIS) -> float:
        prediction = model.predict(X_train)
        residual = y_train - prediction
        if config.objective == "error_std":
            return float(np.std(residual, ddof=1))
        if config.objective == "rmse":
            return float(np.sqrt(np.mean(residual**2)))
        if config.objective == "composite":
            fit_pred = model.predict(X_fit)
            val_pred = model.predict(X_validation)
            fit_rmse = float(np.sqrt(np.mean((y_fit - fit_pred) ** 2)))
            val_rmse = float(np.sqrt(np.mean((y_validation - val_pred) ** 2)))
            jac = model.jacobian(np.vstack([X_fit, X_validation]))
            sensitivity = float(np.mean(np.sum(jac * jac, axis=1)))
            _, activation = model.normalized_weights(X_validation)
            log_shortfall = np.maximum(
                0.0,
                np.log(config.activation_floor)
                - np.log(np.maximum(activation, 1e-300)),
            )
            activation_penalty = float(np.mean(log_shortfall**2))
            return (
                0.5 * fit_rmse
                + val_rmse
                + config.sensitivity_weight * sensitivity
                + config.activation_penalty_weight * activation_penalty
            )
        raise ValueError(f"Unknown objective: {config.objective}")

    if config.protocol == "legacy_intent":
        # Faithful to TrainAnfisUsingPSO.m / TrainFISCost.m intent:
        # PSO searches multiplicative factors in [-25, 25], first particle = ones.
        def cost(factors: np.ndarray) -> float:
            factors = np.asarray(factors, dtype=float).copy()
            tiny = np.abs(factors) < 1e-5
            # Correct the MATLAB sign(0) edge case while preserving intended behavior.
            factors[tiny] = np.where(factors[tiny] < 0.0, -1e-5, 1e-5)
            model = initial_model.from_vector(factors * p0)
            return objective_for_model(model)

        pso_initial = np.ones_like(p0)
        lower, upper = -25.0, 25.0
        pso_config = PSOConfig(
            iterations=config.iterations,
            particles=config.particles,
            inertia=1.0,
            inertia_damping=0.99,
            cognitive=1.0,
            social=2.0,
            seed=config.seed,
            mode="legacy",
        )
        result = particle_swarm_optimize(cost, pso_initial, lower, upper, pso_config)
        final_model = initial_model.from_vector(result.best_position * p0)
    else:
        # Research baseline: PSO optimizes only centers and positive widths.
        # For every candidate, the linear TSK consequents are solved by ridge
        # least squares on the fitting segment (variable-projection strategy).
        antecedent0 = _antecedent_vector(initial_model)
        lower, upper = _make_antecedent_bounds(initial_model, X_fit)

        def cost(antecedents: np.ndarray) -> float:
            candidate = _model_from_antecedents(initial_model, antecedents)
            candidate.fit_consequents(X_fit, y_fit, ridge=1e-4)
            return objective_for_model(candidate)

        pso_config = PSOConfig(
            iterations=config.iterations,
            particles=config.particles,
            seed=config.seed,
            mode="constricted",
        )
        result = particle_swarm_optimize(
            cost, antecedent0, lower, upper, pso_config
        )
        final_model = _model_from_antecedents(initial_model, result.best_position)
        # Once the antecedent geometry has been selected without test access,
        # refit linear consequents using all available training observations.
        final_model.fit_consequents(X_train, y_train, ridge=1e-4)

    train_scaled_pred = final_model.predict(X_train)
    test_scaled_pred = final_model.predict(X_test)
    train_pred = scaler.inverse_transform(train_scaled_pred)
    test_pred = scaler.inverse_transform(test_scaled_pred)

    scale = naive_scale(raw[: config.n_lags + len(ytr_raw)], seasonal_period=1)
    train_metrics = regression_metrics(ytr_raw, train_pred, mase_scale=scale)
    test_metrics = regression_metrics(yte_raw, test_pred, mase_scale=scale)
    initial_test_pred = scaler.inverse_transform(initial_model.predict(X_test))
    initial_test_metrics = regression_metrics(yte_raw, initial_test_pred, mase_scale=scale)
    validation_metrics = None
    if len(X_validation):
        validation_raw = scaler.inverse_transform(y_validation)
        validation_pred_raw = scaler.inverse_transform(final_model.predict(X_validation))
        validation_metrics = regression_metrics(
            validation_raw, validation_pred_raw, mase_scale=scale
        )

    initial_diagnostics = model_diagnostics(initial_model, X_test, seed=config.seed)
    diagnostics_scaled = model_diagnostics(final_model, X_test, seed=config.seed)
    diagnostics = {
        **diagnostics_scaled,
        "sc_selected_rules": int(len(sc_result.centers)),
        "sc_radius": config.sc_radius,
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "internal_validation_samples": int(len(X_validation)),
        "scaler_data_min": scaler.data_min,
        "scaler_data_max": scaler.data_max,
        "best_objective": result.best_cost,
    }

    predictions = pd.DataFrame(
        {
            "date": df["date"].iloc[config.n_lags + len(y_train) :].dt.strftime("%Y-%m-%d"),
            "actual": yte_raw,
            "predicted": test_pred,
            "residual": yte_raw - test_pred,
        }
    )
    predictions.to_csv(out / "test_predictions.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        out / "initial_model.npz",
        centers=initial_model.centers,
        sigmas=initial_model.sigmas,
        consequents=initial_model.consequents,
        scaler_data_min=scaler.data_min,
        scaler_data_max=scaler.data_max,
        scaler_out_min=scaler.out_min,
        scaler_out_max=scaler.out_max,
    )
    np.savez_compressed(
        out / "model.npz",
        centers=final_model.centers,
        sigmas=final_model.sigmas,
        consequents=final_model.consequents,
        scaler_data_min=scaler.data_min,
        scaler_data_max=scaler.data_max,
        scaler_out_min=scaler.out_min,
        scaler_out_max=scaler.out_max,
    )
    pd.DataFrame(result.history).to_csv(out / "pso_history.csv", index=False)
    pd.DataFrame(
        {
            "parameter": np.arange(len(final_model.to_vector())),
            "initial": p0,
            "final": final_model.to_vector(),
        }
    ).to_csv(out / "parameters.csv", index=False)

    summary = {
        "config": asdict(config),
        "initial_test_metrics": initial_test_metrics,
        "initial_diagnostics": initial_diagnostics,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "diagnostics": diagnostics,
        "audit_flags": {
            "full_series_scaling_leakage": config.protocol == "legacy_intent",
            "multiplicative_zero_parameters_are_frozen": config.protocol == "legacy_intent",
            "objective_is_error_std_not_mape": config.objective == "error_std",
        },
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    save_prediction_plot(
        yte_raw,
        test_pred,
        out / "test_actual_vs_predicted.png",
        f"Yemen revenue: {config.protocol}",
    )
    save_residual_plot(
        yte_raw - test_pred,
        out / "test_residuals.png",
        f"Test residuals: {config.protocol}",
    )
    save_pso_history(result.history, out)
    return summary
