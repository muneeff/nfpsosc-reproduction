"""
Forecasting baselines for the SS-NFPSO Q1 experiment pipeline.

The functions in this module implement simple, transparent, leakage-aware
one-step-ahead rolling forecasts. They are intended for fair comparison with
the proposed projected-constricted NFPSO formulation.

Design principles
-----------------
1. Chronological evaluation only.
2. At test step t, models may use y[0:t] but not y[t] or future values.
3. Optional third-party models are imported inside functions so that simple
   baselines work even when experimental dependencies are not installed.
4. Failures are recorded in BaselineResult.failed_steps instead of being hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ArrayLike = Sequence[float] | np.ndarray | pd.Series


@dataclass
class BaselineResult:
    """Container for one rolling-forecast baseline result."""

    model: str
    y_true: np.ndarray
    y_pred: np.ndarray
    test_indices: np.ndarray
    metrics: Dict[str, float]
    failed_steps: list[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def residuals(self) -> np.ndarray:
        """Return y_true - y_pred."""

        return self.y_true - self.y_pred

    def to_frame(self) -> pd.DataFrame:
        """Return predictions in long tabular format."""

        return pd.DataFrame(
            {
                "model": self.model,
                "test_index": self.test_indices,
                "actual": self.y_true,
                "predicted": self.y_pred,
                "residual": self.residuals,
            }
        )


def as_1d_array(series: ArrayLike) -> np.ndarray:
    """Convert input series to a finite 1D float NumPy array."""

    arr = np.asarray(series, dtype=float).reshape(-1)
    if arr.size < 3:
        raise ValueError("series must contain at least three observations")
    if not np.all(np.isfinite(arr)):
        raise ValueError("series contains NaN or infinite values")
    return arr


def resolve_initial_train_size(
    n: int,
    initial_train_size: Optional[int] = None,
    train_fraction: Optional[float] = None,
    min_train_size: int = 10,
) -> int:
    """Resolve the chronological initial training size."""

    if initial_train_size is None:
        if train_fraction is None:
            train_fraction = 0.8
        if not 0 < train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        initial_train_size = int(round(n * float(train_fraction)))

    initial_train_size = int(initial_train_size)
    if initial_train_size < min_train_size:
        raise ValueError(
            f"initial_train_size={initial_train_size} is too small; "
            f"minimum is {min_train_size}"
        )
    if initial_train_size >= n:
        raise ValueError("initial_train_size must be smaller than series length")
    return initial_train_size


def make_lagged_xy(series: ArrayLike, n_lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a univariate series into lagged supervised samples.

    X[i] = [y[t-n_lags], ..., y[t-1]]
    y[i] = y[t]
    """

    y = as_1d_array(series)
    n_lags = int(n_lags)
    if n_lags < 1:
        raise ValueError("n_lags must be positive")
    if len(y) <= n_lags:
        raise ValueError("series length must be greater than n_lags")

    X = []
    target = []
    for t in range(n_lags, len(y)):
        X.append(y[t - n_lags : t])
        target.append(y[t])
    return np.asarray(X, dtype=float), np.asarray(target, dtype=float)


def safe_divide(num: float, den: float, default: float = np.nan) -> float:
    """Safely divide two scalars."""

    if den == 0 or not np.isfinite(den):
        return default
    return float(num / den)


def regression_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    training_series_for_mase: Optional[ArrayLike] = None,
    season_length: int = 1,
) -> Dict[str, float]:
    """Compute common forecasting metrics."""

    actual = as_1d_array(y_true)
    pred = as_1d_array(y_pred)
    if actual.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    err = actual - pred
    abs_err = np.abs(err)
    sq_err = err**2

    mse = float(np.mean(sq_err))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_err))

    nonzero = np.abs(actual) > 1e-12
    if np.any(nonzero):
        mape_fraction = float(np.mean(abs_err[nonzero] / np.abs(actual[nonzero])))
        mape_percent = 100.0 * mape_fraction
    else:
        mape_fraction = np.nan
        mape_percent = np.nan

    smape_den = np.abs(actual) + np.abs(pred)
    smape_terms = np.where(smape_den > 1e-12, 2.0 * abs_err / smape_den, np.nan)
    smape_fraction = float(np.nanmean(smape_terms))
    smape_percent = 100.0 * smape_fraction

    ss_res = float(np.sum(sq_err))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - safe_divide(ss_res, ss_tot, default=np.nan)

    mase = np.nan
    if training_series_for_mase is not None:
        train = as_1d_array(training_series_for_mase)
        lag = max(int(season_length), 1)
        if len(train) > lag:
            naive_scale = float(np.mean(np.abs(train[lag:] - train[:-lag])))
            mase = safe_divide(mae, naive_scale, default=np.nan)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape_fraction": mape_fraction,
        "mape_percent": mape_percent,
        "smape_fraction": smape_fraction,
        "smape_percent": smape_percent,
        "r2": float(r2),
        "mase": float(mase),
        "error_mean": float(np.mean(err)),
        "error_std_sample": float(np.std(err, ddof=1)) if len(err) > 1 else 0.0,
    }


def forecast_naive_1(history: np.ndarray) -> float:
    """One-step persistence forecast."""

    return float(history[-1])


def forecast_seasonal_naive(history: np.ndarray, season_length: int) -> float:
    """Seasonal persistence forecast."""

    season_length = int(season_length)
    if season_length < 1:
        raise ValueError("season_length must be positive")
    if len(history) <= season_length:
        return forecast_naive_1(history)
    return float(history[-season_length])


def forecast_drift(history: np.ndarray) -> float:
    """
    Drift forecast.

    Forecast h=1 ahead using a line from the first to the latest observation.
    """

    if len(history) < 2:
        return forecast_naive_1(history)
    slope = (history[-1] - history[0]) / max(len(history) - 1, 1)
    return float(history[-1] + slope)


def rolling_simple_forecast(
    series: ArrayLike,
    model: str,
    initial_train_size: Optional[int] = None,
    train_fraction: Optional[float] = None,
    season_length: int = 1,
) -> BaselineResult:
    """Run simple deterministic baselines in a rolling one-step protocol."""

    y = as_1d_array(series)
    start = resolve_initial_train_size(
        len(y),
        initial_train_size=initial_train_size,
        train_fraction=train_fraction,
    )

    preds = []
    actuals = []
    test_indices = []

    for t in range(start, len(y)):
        history = y[:t]
        if model == "naive_1":
            pred = forecast_naive_1(history)
        elif model == "seasonal_naive":
            pred = forecast_seasonal_naive(history, season_length=season_length)
        elif model == "drift":
            pred = forecast_drift(history)
        else:
            raise ValueError(f"Unsupported simple model: {model}")

        preds.append(pred)
        actuals.append(float(y[t]))
        test_indices.append(t)

    y_true = np.asarray(actuals, dtype=float)
    y_pred = np.asarray(preds, dtype=float)
    metrics = regression_metrics(
        y_true,
        y_pred,
        training_series_for_mase=y[:start],
        season_length=season_length,
    )
    return BaselineResult(
        model=model,
        y_true=y_true,
        y_pred=y_pred,
        test_indices=np.asarray(test_indices, dtype=int),
        metrics=metrics,
        metadata={
            "initial_train_size": start,
            "season_length": int(season_length),
            "protocol": "rolling_one_step",
        },
    )


def _fallback_prediction(history: np.ndarray, fallback: str, season_length: int) -> float:
    """Return fallback prediction after a model failure."""

    if fallback == "naive_1":
        return forecast_naive_1(history)
    if fallback == "seasonal_naive":
        return forecast_seasonal_naive(history, season_length=season_length)
    if fallback == "drift":
        return forecast_drift(history)
    raise ValueError(f"Unsupported fallback: {fallback}")


def rolling_arima_forecast(
    series: ArrayLike,
    initial_train_size: Optional[int] = None,
    train_fraction: Optional[float] = None,
    order: Tuple[int, int, int] = (1, 1, 0),
    season_length: int = 1,
    fallback: str = "naive_1",
) -> BaselineResult:
    """
    Rolling ARIMA forecast using statsmodels.

    This intentionally uses a fixed order by default. If order selection is
    added later, it must use only the current historical segment at each step.
    """

    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "statsmodels is required for rolling_arima_forecast. "
            "Install requirements-experiments.txt."
        ) from exc

    y = as_1d_array(series)
    start = resolve_initial_train_size(len(y), initial_train_size, train_fraction)

    preds = []
    actuals = []
    test_indices = []
    failed_steps: list[int] = []

    for t in range(start, len(y)):
        history = y[:t]
        try:
            fit = ARIMA(history, order=order).fit()
            pred = float(fit.forecast(steps=1)[0])
            if not np.isfinite(pred):
                raise ValueError("non-finite ARIMA forecast")
        except Exception:
            failed_steps.append(t)
            pred = _fallback_prediction(history, fallback, season_length)

        preds.append(pred)
        actuals.append(float(y[t]))
        test_indices.append(t)

    y_true = np.asarray(actuals, dtype=float)
    y_pred = np.asarray(preds, dtype=float)
    metrics = regression_metrics(y_true, y_pred, y[:start], season_length=season_length)
    return BaselineResult(
        model="arima",
        y_true=y_true,
        y_pred=y_pred,
        test_indices=np.asarray(test_indices, dtype=int),
        metrics=metrics,
        failed_steps=failed_steps,
        metadata={
            "order": tuple(order),
            "fallback": fallback,
            "initial_train_size": start,
            "season_length": int(season_length),
            "protocol": "rolling_one_step",
        },
    )


def rolling_ets_forecast(
    series: ArrayLike,
    initial_train_size: Optional[int] = None,
    train_fraction: Optional[float] = None,
    trend: Optional[str] = "add",
    seasonal: Optional[str] = None,
    season_length: int = 1,
    fallback: str = "naive_1",
) -> BaselineResult:
    """Rolling Exponential Smoothing forecast using statsmodels."""

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "statsmodels is required for rolling_ets_forecast. "
            "Install requirements-experiments.txt."
        ) from exc

    y = as_1d_array(series)
    start = resolve_initial_train_size(len(y), initial_train_size, train_fraction)
    failed_steps: list[int] = []
    preds = []
    actuals = []
    test_indices = []

    for t in range(start, len(y)):
        history = y[:t]
        try:
            seasonal_periods = int(season_length) if seasonal is not None else None
            fit = ExponentialSmoothing(
                history,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            ).fit(optimized=True)
            pred = float(fit.forecast(1)[0])
            if not np.isfinite(pred):
                raise ValueError("non-finite ETS forecast")
        except Exception:
            failed_steps.append(t)
            pred = _fallback_prediction(history, fallback, season_length)

        preds.append(pred)
        actuals.append(float(y[t]))
        test_indices.append(t)

    y_true = np.asarray(actuals, dtype=float)
    y_pred = np.asarray(preds, dtype=float)
    metrics = regression_metrics(y_true, y_pred, y[:start], season_length=season_length)
    return BaselineResult(
        model="ets",
        y_true=y_true,
        y_pred=y_pred,
        test_indices=np.asarray(test_indices, dtype=int),
        metrics=metrics,
        failed_steps=failed_steps,
        metadata={
            "trend": trend,
            "seasonal": seasonal,
            "season_length": int(season_length),
            "fallback": fallback,
            "initial_train_size": start,
            "protocol": "rolling_one_step",
        },
    )


def rolling_theta_forecast(
    series: ArrayLike,
    initial_train_size: Optional[int] = None,
    train_fraction: Optional[float] = None,
    season_length: int = 1,
    fallback: str = "naive_1",
) -> BaselineResult:
    """Rolling Theta-method forecast using statsmodels."""

    try:
        from statsmodels.tsa.forecasting.theta import ThetaModel
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "statsmodels is required for rolling_theta_forecast. "
            "Install requirements-experiments.txt."
        ) from exc

    y = as_1d_array(series)
    start = resolve_initial_train_size(len(y), initial_train_size, train_fraction)
    failed_steps: list[int] = []
    preds = []
    actuals = []
    test_indices = []

    for t in range(start, len(y)):
        history = y[:t]
        try:
            period = int(season_length) if season_length and season_length > 1 else None
            model = ThetaModel(history, period=period)
            fit = model.fit()
            pred = float(fit.forecast(1)[0])
            if not np.isfinite(pred):
                raise ValueError("non-finite Theta forecast")
        except Exception:
            failed_steps.append(t)
            pred = _fallback_prediction(history, fallback, season_length)

        preds.append(pred)
        actuals.append(float(y[t]))
        test_indices.append(t)

    y_true = np.asarray(actuals, dtype=float)
    y_pred = np.asarray(preds, dtype=float)
    metrics = regression_metrics(y_true, y_pred, y[:start], season_length=season_length)
    return BaselineResult(
        model="theta",
        y_true=y_true,
        y_pred=y_pred,
        test_indices=np.asarray(test_indices, dtype=int),
        metrics=metrics,
        failed_steps=failed_steps,
        metadata={
            "season_length": int(season_length),
            "fallback": fallback,
            "initial_train_size": start,
            "protocol": "rolling_one_step",
        },
    )


def _fit_standardized_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model: Any,
) -> tuple[Any, Any]:
    """Fit StandardScaler + regressor using training data only."""

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = make_pipeline(StandardScaler(), model)
    pipeline.fit(X_train, y_train)
    return pipeline, model


def rolling_svr_forecast(
    series: ArrayLike,
    initial_train_size: Optional[int] = None,
    train_fraction: Optional[float] = None,
    n_lags: int = 5,
    season_length: int = 1,
    fallback: str = "naive_1",
    C: float = 10.0,
    epsilon: float = 0.05,
    gamma: str = "scale",
) -> BaselineResult:
    """Rolling SVR forecast with lagged inputs and training-only scaling."""

    try:
        from sklearn.svm import SVR
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required for rolling_svr_forecast. "
            "Install requirements-experiments.txt."
        ) from exc

    y = as_1d_array(series)
    start = resolve_initial_train_size(
        len(y),
        initial_train_size=initial_train_size,
        train_fraction=train_fraction,
        min_train_size=max(10, n_lags + 2),
    )

    failed_steps: list[int] = []
    preds = []
    actuals = []
    test_indices = []

    for t in range(start, len(y)):
        history = y[:t]
        try:
            X_train, y_train = make_lagged_xy(history, n_lags=n_lags)
            x_current = history[-n_lags:].reshape(1, -1)
            estimator = SVR(kernel="rbf", C=float(C), epsilon=float(epsilon), gamma=gamma)
            pipeline, _ = _fit_standardized_regressor(X_train, y_train, estimator)
            pred = float(pipeline.predict(x_current)[0])
            if not np.isfinite(pred):
                raise ValueError("non-finite SVR forecast")
        except Exception:
            failed_steps.append(t)
            pred = _fallback_prediction(history, fallback, season_length)

        preds.append(pred)
        actuals.append(float(y[t]))
        test_indices.append(t)

    y_true = np.asarray(actuals, dtype=float)
    y_pred = np.asarray(preds, dtype=float)
    metrics = regression_metrics(y_true, y_pred, y[:start], season_length=season_length)
    return BaselineResult(
        model="svr",
        y_true=y_true,
        y_pred=y_pred,
        test_indices=np.asarray(test_indices, dtype=int),
        metrics=metrics,
        failed_steps=failed_steps,
        metadata={
            "n_lags": int(n_lags),
            "C": float(C),
            "epsilon": float(epsilon),
            "gamma": gamma,
            "fallback": fallback,
            "initial_train_size": start,
            "protocol": "rolling_one_step",
        },
    )


def rolling_xgboost_forecast(
    series: ArrayLike,
    initial_train_size: Optional[int] = None,
    train_fraction: Optional[float] = None,
    n_lags: int = 5,
    season_length: int = 1,
    fallback: str = "naive_1",
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    subsample: float = 1.0,
    random_state: int = 1,
) -> BaselineResult:
    """Rolling XGBoost forecast with lagged inputs."""

    try:
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "xgboost is required for rolling_xgboost_forecast. "
            "Install requirements-experiments.txt."
        ) from exc

    y = as_1d_array(series)
    start = resolve_initial_train_size(
        len(y),
        initial_train_size=initial_train_size,
        train_fraction=train_fraction,
        min_train_size=max(10, n_lags + 2),
    )

    failed_steps: list[int] = []
    preds = []
    actuals = []
    test_indices = []

    for t in range(start, len(y)):
        history = y[:t]
        try:
            X_train, y_train = make_lagged_xy(history, n_lags=n_lags)
            x_current = history[-n_lags:].reshape(1, -1)
            model = XGBRegressor(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                learning_rate=float(learning_rate),
                subsample=float(subsample),
                objective="reg:squarederror",
                random_state=int(random_state),
                n_jobs=1,
                verbosity=0,
            )
            model.fit(X_train, y_train)
            pred = float(model.predict(x_current)[0])
            if not np.isfinite(pred):
                raise ValueError("non-finite XGBoost forecast")
        except Exception:
            failed_steps.append(t)
            pred = _fallback_prediction(history, fallback, season_length)

        preds.append(pred)
        actuals.append(float(y[t]))
        test_indices.append(t)

    y_true = np.asarray(actuals, dtype=float)
    y_pred = np.asarray(preds, dtype=float)
    metrics = regression_metrics(y_true, y_pred, y[:start], season_length=season_length)
    return BaselineResult(
        model="xgboost",
        y_true=y_true,
        y_pred=y_pred,
        test_indices=np.asarray(test_indices, dtype=int),
        metrics=metrics,
        failed_steps=failed_steps,
        metadata={
            "n_lags": int(n_lags),
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "subsample": float(subsample),
            "random_state": int(random_state),
            "fallback": fallback,
            "initial_train_size": start,
            "protocol": "rolling_one_step",
        },
    )


def run_baseline(
    series: ArrayLike,
    model: str,
    initial_train_size: Optional[int] = None,
    train_fraction: Optional[float] = None,
    n_lags: int = 5,
    season_length: int = 1,
    **kwargs,
) -> BaselineResult:
    """
    Run a named baseline in rolling one-step-ahead mode.

    Supported model names:
    naive_1, seasonal_naive, drift, arima, ets, theta, svr, xgboost.
    """

    model = model.lower().strip()

    if model in {"naive", "naive_1", "persistence"}:
        return rolling_simple_forecast(
            series,
            model="naive_1",
            initial_train_size=initial_train_size,
            train_fraction=train_fraction,
            season_length=season_length,
        )

    if model == "seasonal_naive":
        return rolling_simple_forecast(
            series,
            model="seasonal_naive",
            initial_train_size=initial_train_size,
            train_fraction=train_fraction,
            season_length=season_length,
        )

    if model == "drift":
        return rolling_simple_forecast(
            series,
            model="drift",
            initial_train_size=initial_train_size,
            train_fraction=train_fraction,
            season_length=season_length,
        )

    if model == "arima":
        return rolling_arima_forecast(
            series,
            initial_train_size=initial_train_size,
            train_fraction=train_fraction,
            season_length=season_length,
            **kwargs,
        )

    if model == "ets":
        return rolling_ets_forecast(
            series,
            initial_train_size=initial_train_size,
            train_fraction=train_fraction,
            season_length=season_length,
            **kwargs,
        )

    if model == "theta":
        return rolling_theta_forecast(
            series,
            initial_train_size=initial_train_size,
            train_fraction=train_fraction,
            season_length=season_length,
            **kwargs,
        )

    if model == "svr":
        return rolling_svr_forecast(
            series,
            initial_train_size=initial_train_size,
            train_fraction=train_fraction,
            n_lags=n_lags,
            season_length=season_length,
            **kwargs,
        )

    if model == "xgboost":
        return rolling_xgboost_forecast(
            series,
            initial_train_size=initial_train_size,
            train_fraction=train_fraction,
            n_lags=n_lags,
            season_length=season_length,
            **kwargs,
        )

    raise ValueError(f"Unsupported baseline model: {model}")


def run_many_baselines(
    series: ArrayLike,
    models: Iterable[str],
    initial_train_size: Optional[int] = None,
    train_fraction: Optional[float] = None,
    n_lags: int = 5,
    season_length: int = 1,
    continue_on_error: bool = True,
    **kwargs,
) -> Dict[str, BaselineResult]:
    """
    Run several baselines on the same series.

    Returns a dictionary keyed by model name.
    """

    results: Dict[str, BaselineResult] = {}
    for model_name in models:
        try:
            results[model_name] = run_baseline(
                series,
                model=model_name,
                initial_train_size=initial_train_size,
                train_fraction=train_fraction,
                n_lags=n_lags,
                season_length=season_length,
                **kwargs.get(model_name, {}),
            )
        except Exception:
            if not continue_on_error:
                raise
    return results


def metrics_frame(results: Dict[str, BaselineResult]) -> pd.DataFrame:
    """Convert a dictionary of BaselineResult objects to a metrics DataFrame."""

    rows = []
    for model_name, result in results.items():
        row = {"model": model_name, **result.metrics}
        row["failed_steps"] = len(result.failed_steps)
        row["n_test"] = len(result.y_true)
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "BaselineResult",
    "as_1d_array",
    "make_lagged_xy",
    "regression_metrics",
    "forecast_naive_1",
    "forecast_seasonal_naive",
    "forecast_drift",
    "rolling_simple_forecast",
    "rolling_arima_forecast",
    "rolling_ets_forecast",
    "rolling_theta_forecast",
    "rolling_svr_forecast",
    "rolling_xgboost_forecast",
    "run_baseline",
    "run_many_baselines",
    "metrics_frame",
]
