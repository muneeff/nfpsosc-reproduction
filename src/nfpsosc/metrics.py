from __future__ import annotations

import numpy as np


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mase_scale: float | None = None,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("Metric inputs must be non-empty and equal length.")
    e = y - p
    mse = float(np.mean(e**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(e)))
    valid = np.abs(y) > epsilon
    mape = float(np.mean(np.abs(e[valid] / y[valid]))) if np.any(valid) else float("nan")
    smape = float(np.mean(2.0 * np.abs(e) / (np.abs(y) + np.abs(p) + epsilon)))
    ss_res = float(np.sum(e**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > epsilon else float("nan")
    corr = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 else float("nan")
    result = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape_fraction": mape,
        "mape_percent": 100.0 * mape,
        "smape_fraction": smape,
        "smape_percent": 100.0 * smape,
        "r2": r2,
        "pearson_r": corr,
        "error_mean": float(np.mean(e)),
        "error_std_sample": float(np.std(e, ddof=1)) if len(e) > 1 else 0.0,
    }
    if mase_scale is not None:
        result["mase"] = mae / max(float(mase_scale), epsilon)
    return result


def naive_scale(training_series: np.ndarray, seasonal_period: int = 1) -> float:
    y = np.asarray(training_series, dtype=float).reshape(-1)
    if seasonal_period < 1 or len(y) <= seasonal_period:
        raise ValueError("Invalid seasonal period for MASE scale.")
    return float(np.mean(np.abs(y[seasonal_period:] - y[:-seasonal_period])))
