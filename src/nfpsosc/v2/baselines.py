from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def first_forecast_value(forecast) -> float:
    """
    Return the first forecast value by POSITION.

    V2 explicitly prohibits pandas label access such as forecast[0].
    """
    if isinstance(forecast, pd.Series):
        if len(forecast) < 1:
            raise ValueError("forecast is empty")

        value = float(forecast.iloc[0])

    else:
        array = np.asarray(
            forecast,
            dtype=float,
        ).reshape(-1)

        if array.size < 1:
            raise ValueError("forecast is empty")

        value = float(array[0])

    if not np.isfinite(value):
        raise ValueError(
            "forecast contains a non-finite first value"
        )

    return value


@dataclass
class FrozenTheta:
    """
    Theta state for the frozen V2 B1 test policy.

    Parameters b0 and alpha are estimated exactly once on the
    complete pre-test training segment.

    During the test window:
      - b0 is never re-estimated;
      - alpha is never re-estimated;
      - the fixed additive seasonal cycle is not re-estimated;
      - only the SES state is updated from newly observed truth;
      - nobs advances by one after each observation.
    """

    b0: float
    alpha: float
    ses_state: float
    nobs: int
    seasonal_cycle: np.ndarray
    seasonal_period: int
    seasonal_position: int = 0

    @classmethod
    def fit(
        cls,
        training_series,
        seasonal_period: int = 1,
    ) -> "FrozenTheta":
        from statsmodels.tsa.forecasting.theta import (
            ThetaModel,
        )

        if isinstance(training_series, pd.Series):
            y = training_series.astype(float).copy()
        else:
            values = np.asarray(
                training_series,
                dtype=float,
            ).reshape(-1)

            y = pd.Series(values)

        if len(y) < 3:
            raise ValueError(
                "Theta training series must contain at least "
                "three observations"
            )

        if not np.all(
            np.isfinite(
                y.to_numpy(dtype=float)
            )
        ):
            raise ValueError(
                "Theta training series contains non-finite values"
            )

        m = int(seasonal_period)

        if m < 1:
            raise ValueError(
                "seasonal_period must be >= 1"
            )

        if m > 1:
            model = ThetaModel(
                y,
                period=m,
                deseasonalize=True,
                use_test=False,
                method="additive",
            )
        else:
            model = ThetaModel(
                y,
                period=None,
                deseasonalize=False,
                use_test=False,
                method="additive",
            )

        result = model.fit(
            use_mle=False,
            disp=False,
        )

        params = result.params

        b0 = float(params["b0"])
        alpha = float(params["alpha"])

        if not np.isfinite(b0):
            raise ValueError(
                "Theta b0 is non-finite"
            )

        if not np.isfinite(alpha):
            raise ValueError(
                "Theta alpha is non-finite"
            )

        # Use the PUBLIC forecast-components API to recover the
        # SES one-step state. Do not depend on private _one_step.
        component_steps = m if m > 1 else 1

        components = result.forecast_components(
            component_steps
        )

        ses_state = float(
            components["ses"].iloc[0]
        )

        if not np.isfinite(ses_state):
            raise ValueError(
                "Theta SES state is non-finite"
            )

        if m > 1:
            seasonal_cycle = (
                components["seasonal"]
                .to_numpy(dtype=float)
                .copy()
            )

            if seasonal_cycle.size != m:
                raise ValueError(
                    "Theta seasonal cycle has unexpected length"
                )
        else:
            seasonal_cycle = np.zeros(
                1,
                dtype=float,
            )

        if not np.all(
            np.isfinite(seasonal_cycle)
        ):
            raise ValueError(
                "Theta seasonal cycle contains non-finite values"
            )

        return cls(
            b0=b0,
            alpha=alpha,
            ses_state=ses_state,
            nobs=len(y),
            seasonal_cycle=seasonal_cycle,
            seasonal_period=m,
            seasonal_position=0,
        )

    def current_seasonal(self) -> float:
        if self.seasonal_period <= 1:
            return 0.0

        index = (
            self.seasonal_position
            % self.seasonal_period
        )

        return float(
            self.seasonal_cycle[index]
        )

    def _trend_component(self) -> float:
        """
        statsmodels Theta one-step trend component for theta=2.

        For h=1, h-1 = 0.
        """
        if self.alpha > 0.0:
            h_term = (
                1.0 / self.alpha
                - (
                    (1.0 - self.alpha)
                    ** self.nobs
                )
                / self.alpha
            )
        else:
            h_term = 0.0

        return float(
            self.b0 * h_term
        )

    def forecast_one(self) -> float:
        """
        One-step theta=2 forecast using frozen model parameters.
        """
        trend = self._trend_component()

        # statsmodels:
        # trend_weight=(theta-1)/theta=0.5 for theta=2.
        forecast = (
            0.5 * trend
            + self.ses_state
            + self.current_seasonal()
        )

        forecast = float(forecast)

        if not np.isfinite(forecast):
            raise ValueError(
                "Theta forecast is non-finite"
            )

        return forecast

    def observe(
        self,
        observed_value: float,
    ) -> None:
        """
        Update only the SES state using the observed true value.

        No b0, alpha, or seasonality estimation occurs here.
        """
        observed = float(observed_value)

        if not np.isfinite(observed):
            raise ValueError(
                "observed value must be finite"
            )

        seasonal = self.current_seasonal()

        deseasonalized = (
            observed - seasonal
        )

        self.ses_state = float(
            self.alpha * deseasonalized
            + (1.0 - self.alpha)
            * self.ses_state
        )

        self.nobs += 1

        if self.seasonal_period > 1:
            self.seasonal_position = (
                self.seasonal_position + 1
            ) % self.seasonal_period
@dataclass
class FrozenSARIMA:
    """
    SARIMA state under the frozen V2 B1 policy.

    Model parameters are estimated exactly once on the complete
    pre-test training segment.

    During the test window, observed truth is appended with
    refit=False so that latent states may update while fitted
    parameters remain unchanged.
    """

    _result: object
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    trend: str | None
    _frozen_params: np.ndarray

    @classmethod
    def fit(
        cls,
        training_series,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int] = (
            0,
            0,
            0,
            0,
        ),
        trend: str | None = None,
    ) -> "FrozenSARIMA":
        from statsmodels.tsa.statespace.sarimax import (
            SARIMAX,
        )

        y = np.asarray(
            training_series,
            dtype=float,
        ).reshape(-1)

        if y.size < 3:
            raise ValueError(
                "SARIMA training series must contain at least "
                "three observations"
            )

        if not np.all(np.isfinite(y)):
            raise ValueError(
                "SARIMA training series contains non-finite values"
            )

        result = SARIMAX(
            y,
            order=tuple(order),
            seasonal_order=tuple(seasonal_order),
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(
            disp=False,
        )

        params = np.asarray(
            result.params,
            dtype=float,
        ).copy()

        if not np.all(np.isfinite(params)):
            raise ValueError(
                "SARIMA fitted parameters contain non-finite values"
            )

        return cls(
            _result=result,
            order=tuple(order),
            seasonal_order=tuple(seasonal_order),
            trend=trend,
            _frozen_params=params,
        )

    @property
    def params(self) -> np.ndarray:
        return np.asarray(
            self._result.params,
            dtype=float,
        ).copy()

    @property
    def nobs(self) -> int:
        return int(self._result.nobs)

    @property
    def aicc(self) -> float:
        return float(self._result.aicc)

    @property
    def converged(self) -> bool:
        """
        True only when statsmodels explicitly reports successful
        maximum-likelihood convergence.
        """
        retvals = getattr(
            self._result,
            "mle_retvals",
            None,
        )

        if not isinstance(retvals, dict):
            return False

        flag = retvals.get(
            "converged",
            None,
        )

        if not isinstance(
            flag,
            (bool, np.bool_),
        ):
            return False

        return bool(flag)

    def forecast_one(self) -> float:
        forecast = self._result.forecast(
            steps=1
        )

        return first_forecast_value(
            forecast
        )

    def observe(
        self,
        observed_value: float,
    ) -> None:
        observed = float(observed_value)

        if not np.isfinite(observed):
            raise ValueError(
                "observed value must be finite"
            )

        params_before = self.params

        # Critical V2 B1 rule:
        # update state with observed truth but NEVER refit parameters.
        self._result = self._result.append(
            np.asarray([observed], dtype=float),
            refit=False,
        )

        params_after = self.params

        # Runtime invariant against accidental test-time refitting
        # or parameter mutation.
        if not np.array_equal(
            params_before,
            params_after,
        ):
            raise RuntimeError(
                "SARIMA parameters changed during "
                "test-time state update"
            )

        if not np.array_equal(
            self._frozen_params,
            params_after,
        ):
            raise RuntimeError(
                "SARIMA parameters differ from frozen "
                "pre-test parameters"
            )
@dataclass
class FrozenETS:
    """
    Holt-Winters ETS execution under frozen V2 B1 policy.

    Parameters and initialization are estimated once from the
    complete pre-test training segment.

    During testing, observed truth is incorporated by replaying
    the deterministic Holt-Winters recursions using the frozen
    parameters and frozen initial states with optimized=False.
    """

    history: np.ndarray

    trend: str | None
    damped_trend: bool
    seasonal: str | None
    seasonal_periods: int | None

    smoothing_level: float
    smoothing_trend: float | None
    smoothing_seasonal: float | None
    damping_trend: float | None

    initial_level: float
    initial_trend: float | None
    initial_seasons: np.ndarray | None

    selection_aicc: float
    optimizer_success: bool

    @classmethod
    def fit(
        cls,
        training_series,
        trend: str | None,
        damped_trend: bool,
        seasonal: str | None,
        seasonal_periods: int | None,
    ) -> "FrozenETS":
        from statsmodels.tsa.holtwinters import (
            ExponentialSmoothing,
        )

        y = np.asarray(
            training_series,
            dtype=float,
        ).reshape(-1)

        if y.size < 3:
            raise ValueError(
                "ETS training series must contain at least "
                "three observations"
            )

        if not np.all(np.isfinite(y)):
            raise ValueError(
                "ETS training series contains non-finite values"
            )

        if trend is None and damped_trend:
            raise ValueError(
                "damped_trend=True requires a trend component"
            )

        if seasonal is None:
            m = None
        else:
            if seasonal_periods is None:
                raise ValueError(
                    "seasonal_periods is required for seasonal ETS"
                )

            m = int(seasonal_periods)

            if m <= 1:
                raise ValueError(
                    "seasonal_periods must be > 1 for seasonal ETS"
                )

        result = ExponentialSmoothing(
            y,
            trend=trend,
            damped_trend=bool(damped_trend),
            seasonal=seasonal,
            seasonal_periods=m,
            initialization_method="estimated",
            use_boxcox=False,
        ).fit(
            optimized=True,
            remove_bias=False,
        )

        params = result.params

        alpha = float(
            params["smoothing_level"]
        )

        beta = (
            float(params["smoothing_trend"])
            if trend is not None
            else None
        )

        gamma = (
            float(params["smoothing_seasonal"])
            if seasonal is not None
            else None
        )

        phi = (
            float(params["damping_trend"])
            if damped_trend
            else None
        )

        initial_level = float(
            params["initial_level"]
        )

        initial_trend = (
            float(params["initial_trend"])
            if trend is not None
            else None
        )

        initial_seasons = (
            np.asarray(
                params["initial_seasons"],
                dtype=float,
            ).copy()
            if seasonal is not None
            else None
        )

        scalar_values = [
            alpha,
            initial_level,
        ]

        if beta is not None:
            scalar_values.append(beta)

        if gamma is not None:
            scalar_values.append(gamma)

        if phi is not None:
            scalar_values.append(phi)

        if initial_trend is not None:
            scalar_values.append(initial_trend)

        if not np.all(
            np.isfinite(
                np.asarray(
                    scalar_values,
                    dtype=float,
                )
            )
        ):
            raise ValueError(
                "ETS frozen parameters contain non-finite values"
            )

        if (
            initial_seasons is not None
            and not np.all(
                np.isfinite(initial_seasons)
            )
        ):
            raise ValueError(
                "ETS initial seasonal states contain "
                "non-finite values"
            )

        aicc = float(result.aicc)

        if not np.isfinite(aicc):
            raise ValueError(
                "ETS fitted AICc is non-finite"
            )

        mle_retvals = getattr(
            result,
            "mle_retvals",
            None,
        )

        optimizer_success = False

        if mle_retvals is not None:
            success = getattr(
                mle_retvals,
                "success",
                None,
            )

            if (
                success is None
                and isinstance(
                    mle_retvals,
                    dict,
                )
            ):
                success = mle_retvals.get(
                    "success",
                    None,
                )

            if isinstance(
                success,
                (bool, np.bool_),
            ):
                optimizer_success = bool(
                    success
                )

        return cls(
            history=y.copy(),
            trend=trend,
            damped_trend=bool(damped_trend),
            seasonal=seasonal,
            seasonal_periods=m,
            smoothing_level=alpha,
            smoothing_trend=beta,
            smoothing_seasonal=gamma,
            damping_trend=phi,
            initial_level=initial_level,
            initial_trend=initial_trend,
            initial_seasons=initial_seasons,
            selection_aicc=aicc,
            optimizer_success=optimizer_success,
        )

    @property
    def nobs(self) -> int:
        return int(
            self.history.size
        )

    @property
    def aicc(self) -> float:
        return float(
            self.selection_aicc
        )

    def frozen_parameters(self) -> dict:
        return {
            "smoothing_level": self.smoothing_level,
            "smoothing_trend": self.smoothing_trend,
            "smoothing_seasonal": self.smoothing_seasonal,
            "damping_trend": self.damping_trend,
            "initial_level": self.initial_level,
            "initial_trend": self.initial_trend,
            "initial_seasons": (
                None
                if self.initial_seasons is None
                else self.initial_seasons.copy()
            ),
        }

    def _replay_result(self):
        from statsmodels.tsa.holtwinters import (
            ExponentialSmoothing,
        )

        kwargs = {
            "trend": self.trend,
            "damped_trend": self.damped_trend,
            "seasonal": self.seasonal,
            "seasonal_periods": self.seasonal_periods,
            "initialization_method": "known",
            "initial_level": self.initial_level,
            "use_boxcox": False,
        }

        if self.trend is not None:
            kwargs["initial_trend"] = (
                self.initial_trend
            )

        if self.seasonal is not None:
            kwargs["initial_seasonal"] = (
                self.initial_seasons.copy()
            )

        model = ExponentialSmoothing(
            self.history,
            **kwargs,
        )

        fit_kwargs = {
            "smoothing_level": self.smoothing_level,
            "optimized": False,
            "remove_bias": False,
        }

        if self.trend is not None:
            fit_kwargs["smoothing_trend"] = (
                self.smoothing_trend
            )

        if self.seasonal is not None:
            fit_kwargs["smoothing_seasonal"] = (
                self.smoothing_seasonal
            )

        if self.damped_trend:
            fit_kwargs["damping_trend"] = (
                self.damping_trend
            )

        result = model.fit(
            **fit_kwargs,
        )

        # statsmodels may represent the no-optimization mask as None
        # rather than the literal bool False.  The robust invariant is that
        # there are no optimizer return values and no parameter is marked as
        # optimized.
        if result.mle_retvals is not None:
            raise RuntimeError(
                "ETS state replay unexpectedly produced optimizer results"
            )

        if bool(np.any(result.optimized)):
            raise RuntimeError(
                "ETS state replay unexpectedly marked parameters as optimized"
            )

        return result

    def forecast_one(self) -> float:
        result = self._replay_result()

        forecast = result.forecast(
            1
        )

        return first_forecast_value(
            forecast
        )

    def observe(
        self,
        observed_value: float,
    ) -> None:
        observed = float(
            observed_value
        )

        if not np.isfinite(observed):
            raise ValueError(
                "observed value must be finite"
            )

        self.history = np.append(
            self.history,
            observed,
        )
from itertools import product


def _lagged_xy_raw(
    series: np.ndarray,
    n_lags: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(series, dtype=float).reshape(-1)
    L = int(n_lags)

    if L < 1 or y.size <= L:
        raise ValueError("invalid lag configuration")

    X = np.asarray(
        [y[t - L:t] for t in range(L, y.size)],
        dtype=float,
    )
    target = y[L:].copy()

    return X, target


def inner_validation_size(
    n_train: int,
    n_lags: int,
) -> int:
    n = int(n_train)
    L = int(n_lags)

    V = min(
        max(5, int(np.ceil(0.20 * n))),
        n - L - 10,
    )

    if V < 5:
        raise ValueError(
            "insufficient training data for frozen "
            "inner-validation rule"
        )

    return int(V)


def _mase_scale(
    training_series,
    seasonal_period: int,
) -> float:
    y = np.asarray(
        training_series,
        dtype=float,
    ).reshape(-1)

    m = max(int(seasonal_period), 1)

    if y.size <= m:
        raise ValueError(
            "insufficient data for MASE denominator"
        )

    scale = float(
        np.mean(
            np.abs(
                y[m:] - y[:-m]
            )
        )
    )

    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError(
            "invalid MASE denominator"
        )

    return scale


@dataclass
class FrozenLagRegressor:
    history: np.ndarray
    n_lags: int
    estimator: object
    x_scaler: object | None
    y_scaler: object | None
    model_name: str

    @classmethod
    def fit_ridge(
        cls,
        training_series,
        n_lags: int,
        alpha: float,
    ) -> "FrozenLagRegressor":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        history = np.asarray(
            training_series,
            dtype=float,
        ).reshape(-1)

        X, y = _lagged_xy_raw(
            history,
            n_lags,
        )

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        Xs = x_scaler.fit_transform(X)
        ys = y_scaler.fit_transform(
            y.reshape(-1, 1)
        ).reshape(-1)

        estimator = Ridge(
            alpha=float(alpha),
        )
        estimator.fit(Xs, ys)

        return cls(
            history=history.copy(),
            n_lags=int(n_lags),
            estimator=estimator,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            model_name="ridge_lag",
        )

    @classmethod
    def fit_svr(
        cls,
        training_series,
        n_lags: int,
        C: float,
        epsilon: float,
        gamma,
    ) -> "FrozenLagRegressor":
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        history = np.asarray(
            training_series,
            dtype=float,
        ).reshape(-1)

        X, y = _lagged_xy_raw(
            history,
            n_lags,
        )

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        Xs = x_scaler.fit_transform(X)
        ys = y_scaler.fit_transform(
            y.reshape(-1, 1)
        ).reshape(-1)

        estimator = SVR(
            kernel="rbf",
            C=float(C),
            epsilon=float(epsilon),
            gamma=gamma,
        )
        estimator.fit(Xs, ys)

        return cls(
            history=history.copy(),
            n_lags=int(n_lags),
            estimator=estimator,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            model_name="svr_rbf",
        )

    @classmethod
    def fit_xgboost(
        cls,
        training_series,
        n_lags: int,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
        min_child_weight: int,
    ) -> "FrozenLagRegressor":
        from xgboost import XGBRegressor

        history = np.asarray(
            training_series,
            dtype=float,
        ).reshape(-1)

        X, y = _lagged_xy_raw(
            history,
            n_lags,
        )

        estimator = XGBRegressor(
            objective="reg:squarederror",
            booster="gbtree",
            tree_method="hist",
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            min_child_weight=int(min_child_weight),
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            reg_alpha=0.0,
            n_jobs=1,
            random_state=271828,
            verbosity=0,
        )

        estimator.fit(X, y)

        return cls(
            history=history.copy(),
            n_lags=int(n_lags),
            estimator=estimator,
            x_scaler=None,
            y_scaler=None,
            model_name="xgboost",
        )

    @property
    def nobs(self) -> int:
        return int(self.history.size)

    def forecast_one(self) -> float:
        if self.history.size < self.n_lags:
            raise ValueError(
                "insufficient observed history"
            )

        x = self.history[
            -self.n_lags:
        ].reshape(1, -1)

        if self.x_scaler is not None:
            x = self.x_scaler.transform(x)

        pred = np.asarray(
            self.estimator.predict(x),
            dtype=float,
        ).reshape(-1)

        if pred.size != 1:
            raise ValueError(
                "lag regressor returned invalid "
                "forecast length"
            )

        value = float(pred[0])

        if self.y_scaler is not None:
            value = float(
                self.y_scaler.inverse_transform(
                    np.asarray([[value]])
                )[0, 0]
            )

        if not np.isfinite(value):
            raise ValueError(
                "lag regressor forecast is non-finite"
            )

        return value

    def observe(
        self,
        observed_value: float,
    ) -> None:
        observed = float(observed_value)

        if not np.isfinite(observed):
            raise ValueError(
                "observed value must be finite"
            )

        # Critical B1 rule:
        # append truth to lag history only.
        # estimator/scalers are NEVER refitted.
        self.history = np.append(
            self.history,
            observed,
        )


def _validation_mase_for_model(
    model: FrozenLagRegressor,
    validation_truth: np.ndarray,
    inner_fit_series: np.ndarray,
    seasonal_period: int,
) -> float:
    preds = []

    for actual in validation_truth:
        preds.append(
            model.forecast_one()
        )
        model.observe(float(actual))

    pred = np.asarray(preds, dtype=float)
    actual = np.asarray(
        validation_truth,
        dtype=float,
    )

    if pred.shape != actual.shape:
        raise ValueError(
            "validation forecast length mismatch"
        )

    if not np.all(np.isfinite(pred)):
        raise ValueError(
            "validation forecasts contain non-finite values"
        )

    scale = _mase_scale(
        inner_fit_series,
        seasonal_period,
    )

    return float(
        np.mean(np.abs(actual - pred))
        / scale
    )


@dataclass(frozen=True)
class RidgeSelection:
    alpha: float
    validation_mase: float


@dataclass(frozen=True)
class SVRSelection:
    C: float
    epsilon: float
    gamma: object
    validation_mase: float


@dataclass(frozen=True)
class XGBSelection:
    n_estimators: int
    max_depth: int
    learning_rate: float
    min_child_weight: int
    validation_mase: float


def _inner_split(
    training_series,
    n_lags: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(
        training_series,
        dtype=float,
    ).reshape(-1)

    V = inner_validation_size(
        y.size,
        n_lags,
    )

    inner_fit = y[:-V]
    validation = y[-V:]

    if inner_fit.size - int(n_lags) < 10:
        raise ValueError(
            "inner-fit has fewer than 10 supervised rows"
        )

    return inner_fit, validation


def select_ridge(
    training_series,
    n_lags: int,
    seasonal_period: int,
    alphas,
) -> RidgeSelection:
    inner_fit, validation = _inner_split(
        training_series,
        n_lags,
    )

    scored = []

    for alpha in alphas:
        model = FrozenLagRegressor.fit_ridge(
            inner_fit,
            n_lags=n_lags,
            alpha=float(alpha),
        )

        score = _validation_mase_for_model(
            model,
            validation,
            inner_fit,
            seasonal_period,
        )

        scored.append(
            (score, -float(alpha), float(alpha))
        )

    # minimum MASE; exact tie -> larger alpha.
    best = min(scored)

    return RidgeSelection(
        alpha=best[2],
        validation_mase=best[0],
    )


def select_svr(
    training_series,
    n_lags: int,
    seasonal_period: int,
    C_values,
    epsilon_values,
    gamma_values,
) -> SVRSelection:
    inner_fit, validation = _inner_split(
        training_series,
        n_lags,
    )

    scored = []

    for C, epsilon, gamma in product(
        C_values,
        epsilon_values,
        gamma_values,
    ):
        model = FrozenLagRegressor.fit_svr(
            inner_fit,
            n_lags=n_lags,
            C=C,
            epsilon=epsilon,
            gamma=gamma,
        )

        score = _validation_mase_for_model(
            model,
            validation,
            inner_fit,
            seasonal_period,
        )

        gamma_priority = (
            0
            if gamma == "scale"
            else 1
        )

        # Frozen tie order:
        # smaller C -> larger epsilon ->
        # scale before numeric gamma.
        scored.append(
            (
                score,
                float(C),
                -float(epsilon),
                gamma_priority,
                C,
                epsilon,
                gamma,
            )
        )

    best = min(
        scored,
        key=lambda row: row[:4],
    )

    return SVRSelection(
        C=float(best[4]),
        epsilon=float(best[5]),
        gamma=best[6],
        validation_mase=float(best[0]),
    )


def select_xgboost(
    training_series,
    n_lags: int,
    seasonal_period: int,
    n_estimators_values,
    max_depth_values,
    learning_rate_values,
    min_child_weight_values,
) -> XGBSelection:
    inner_fit, validation = _inner_split(
        training_series,
        n_lags,
    )

    scored = []

    for (
        n_estimators,
        max_depth,
        learning_rate,
        min_child_weight,
    ) in product(
        n_estimators_values,
        max_depth_values,
        learning_rate_values,
        min_child_weight_values,
    ):
        model = FrozenLagRegressor.fit_xgboost(
            inner_fit,
            n_lags=n_lags,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
        )

        score = _validation_mase_for_model(
            model,
            validation,
            inner_fit,
            seasonal_period,
        )

        scored.append(
            (
                score,
                int(n_estimators),
                int(max_depth),
                float(learning_rate),
                -int(min_child_weight),
            )
        )

    # Frozen tie order:
    # fewer trees, shallower depth,
    # lower learning rate, larger min_child_weight.
    best = min(scored)

    return XGBSelection(
        n_estimators=int(best[1]),
        max_depth=int(best[2]),
        learning_rate=float(best[3]),
        min_child_weight=int(-best[4]),
        validation_mase=float(best[0]),
    )
@dataclass(frozen=True)
class SARIMACandidate:
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    trend: str | None


@dataclass(frozen=True)
class ETSCandidate:
    trend: str | None
    damped_trend: bool
    seasonal: str | None
    seasonal_periods: int | None


@dataclass(frozen=True)
class CandidateFailure:
    model: str
    candidate: object
    error_type: str
    message: str


@dataclass
class SARIMAAICcSelection:
    candidate: SARIMACandidate
    aicc: float
    model: object
    failures: list[CandidateFailure]


@dataclass
class ETSAICcSelection:
    candidate: ETSCandidate
    aicc: float
    model: object
    failures: list[CandidateFailure]


class CandidateValidationError(RuntimeError):
    """Candidate fit completed but is numerically invalid."""


class BaselineSelectionError(RuntimeError):
    def __init__(
        self,
        model_name: str,
        failures: list[CandidateFailure],
    ):
        self.model_name = str(model_name)
        self.failures = list(failures)

        super().__init__(
            f"{self.model_name}: all candidate models failed; "
            "no fallback permitted"
        )


def generate_sarima_candidates(
    seasonal_period: int,
) -> list[SARIMACandidate]:
    m = int(seasonal_period)

    if m < 1:
        raise ValueError(
            "seasonal_period must be >= 1"
        )

    candidates = []

    p_values = (0, 1, 2)
    d_values = (0, 1)
    q_values = (0, 1, 2)

    if m == 1:
        seasonal_terms = [
            (0, 0, 0, 0)
        ]
    else:
        seasonal_terms = [
            (P, D, Q, m)
            for P in (0, 1)
            for D in (0, 1)
            for Q in (0, 1)
        ]

    for p in p_values:
        for d in d_values:
            for q in q_values:
                for seasonal_order in seasonal_terms:
                    P, D, Q, _ = seasonal_order

                    if (
                        p + q + P + Q
                        > 4
                    ):
                        continue

                    # Frozen protocol:
                    # constant is permitted only if
                    # d == 0 and D == 0.
                    trends = (
                        (None, "c")
                        if d == 0 and D == 0
                        else (None,)
                    )

                    for trend in trends:
                        candidates.append(
                            SARIMACandidate(
                                order=(
                                    p,
                                    d,
                                    q,
                                ),
                                seasonal_order=(
                                    seasonal_order
                                ),
                                trend=trend,
                            )
                        )

    return candidates


def sarima_tie_key(
    candidate: SARIMACandidate,
) -> tuple:
    """
    V2-A001 exact AICc tie-break.

    1 smaller p+q+P+Q
    2 smaller P+Q
    3 smaller p+q
    4 smaller d+D
    5 smaller D
    6 lexicographic
      (p,d,q,P,D,Q,trend_code)
    """
    p, d, q = candidate.order
    P, D, Q, _ = candidate.seasonal_order

    trend_code = (
        0
        if candidate.trend is None
        else 1
    )

    return (
        p + q + P + Q,
        P + Q,
        p + q,
        d + D,
        D,
        p,
        d,
        q,
        P,
        D,
        Q,
        trend_code,
    )


def generate_ets_candidates(
    seasonal_period: int,
) -> list[ETSCandidate]:
    m = int(seasonal_period)

    if m < 1:
        raise ValueError(
            "seasonal_period must be >= 1"
        )

    seasonal_values = (
        (None,)
        if m == 1
        else (None, "add")
    )

    candidates = []

    for seasonal in seasonal_values:
        seasonal_periods = (
            None
            if seasonal is None
            else m
        )

        # No trend -> damping is prohibited.
        candidates.append(
            ETSCandidate(
                trend=None,
                damped_trend=False,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
            )
        )

        # Additive trend:
        # both undamped and damped are allowed.
        for damped in (
            False,
            True,
        ):
            candidates.append(
                ETSCandidate(
                    trend="add",
                    damped_trend=damped,
                    seasonal=seasonal,
                    seasonal_periods=seasonal_periods,
                )
            )

    return candidates


def ets_tie_key(
    candidate: ETSCandidate,
) -> tuple:
    """
    V2-A001 exact AICc tie-break.
    """
    trend_code = int(
        candidate.trend == "add"
    )

    seasonal_code = int(
        candidate.seasonal == "add"
    )

    component_count = (
        trend_code
        + seasonal_code
    )

    damped_code = int(
        candidate.damped_trend
    )

    return (
        component_count,
        damped_code,
        seasonal_code,
        trend_code,
    )


def _fit_sarima_candidate(
    training_series,
    candidate: SARIMACandidate,
):
    return FrozenSARIMA.fit(
        training_series,
        order=candidate.order,
        seasonal_order=(
            candidate.seasonal_order
        ),
        trend=candidate.trend,
    )


def _fit_ets_candidate(
    training_series,
    candidate: ETSCandidate,
):
    return FrozenETS.fit(
        training_series,
        trend=candidate.trend,
        damped_trend=(
            candidate.damped_trend
        ),
        seasonal=candidate.seasonal,
        seasonal_periods=(
            candidate.seasonal_periods
        ),
    )


def select_sarima_aicc(
    training_series,
    seasonal_period: int,
) -> SARIMAAICcSelection:
    candidates = generate_sarima_candidates(
        seasonal_period
    )

    failures = []
    successes = []

    for candidate in candidates:
        try:
            import warnings

            from statsmodels.tools.sm_exceptions import (
                ConvergenceWarning,
            )

            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore",
                    ConvergenceWarning,
                )

                model = _fit_sarima_candidate(
                    training_series,
                    candidate,
                )

            if not bool(
                getattr(
                    model,
                    "converged",
                    False,
                )
            ):
                raise CandidateValidationError(
                    "SARIMA candidate did not converge"
                )

            aicc = float(
                model.aicc
            )

            if not np.isfinite(aicc):
                raise CandidateValidationError(
                    "non-finite SARIMA AICc"
                )

            successes.append(
                (
                    aicc,
                    sarima_tie_key(
                        candidate
                    ),
                    candidate,
                    model,
                )
            )

        except Exception as exc:
            failures.append(
                CandidateFailure(
                    model="sarima",
                    candidate=candidate,
                    error_type=(
                        type(exc).__name__
                    ),
                    message=str(exc),
                )
            )

    if not successes:
        raise BaselineSelectionError(
            "SARIMA",
            failures,
        )

    # Exact numeric AICc comparison first.
    # Tie-break is consulted only when
    # the finite AICc values are equal.
    best = min(
        successes,
        key=lambda row: (
            row[0],
            row[1],
        ),
    )

    return SARIMAAICcSelection(
        candidate=best[2],
        aicc=float(best[0]),
        model=best[3],
        failures=failures,
    )


def select_ets_aicc(
    training_series,
    seasonal_period: int,
) -> ETSAICcSelection:
    candidates = generate_ets_candidates(
        seasonal_period
    )

    failures = []
    successes = []

    for candidate in candidates:
        try:
            import warnings

            from statsmodels.tools.sm_exceptions import (
                ConvergenceWarning,
            )

            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore",
                    ConvergenceWarning,
                )

                model = _fit_ets_candidate(
                    training_series,
                    candidate,
                )

            if not bool(
                getattr(
                    model,
                    "optimizer_success",
                    False,
                )
            ):
                raise CandidateValidationError(
                    "ETS optimizer did not converge"
                )

            aicc = float(
                model.aicc
            )

            if not np.isfinite(aicc):
                raise CandidateValidationError(
                    "non-finite ETS AICc"
                )

            successes.append(
                (
                    aicc,
                    ets_tie_key(
                        candidate
                    ),
                    candidate,
                    model,
                )
            )

        except Exception as exc:
            failures.append(
                CandidateFailure(
                    model="ets",
                    candidate=candidate,
                    error_type=(
                        type(exc).__name__
                    ),
                    message=str(exc),
                )
            )

    if not successes:
        raise BaselineSelectionError(
            "ETS",
            failures,
        )

    best = min(
        successes,
        key=lambda row: (
            row[0],
            row[1],
        ),
    )

    return ETSAICcSelection(
        candidate=best[2],
        aicc=float(best[0]),
        model=best[3],
        failures=failures,
    )
@dataclass
class FrozenNaive1:
    last_value: float
    nobs: int

    @classmethod
    def fit(
        cls,
        training_series,
    ) -> "FrozenNaive1":
        y = np.asarray(
            training_series,
            dtype=float,
        ).reshape(-1)

        if y.size < 1:
            raise ValueError(
                "naive_1 requires non-empty training data"
            )

        if not np.all(np.isfinite(y)):
            raise ValueError(
                "naive_1 training series contains "
                "non-finite values"
            )

        return cls(
            last_value=float(y[-1]),
            nobs=int(y.size),
        )

    def forecast_one(self) -> float:
        return float(
            self.last_value
        )

    def observe(
        self,
        observed_value: float,
    ) -> None:
        observed = float(
            observed_value
        )

        if not np.isfinite(observed):
            raise ValueError(
                "observed value must be finite"
            )

        self.last_value = observed
        self.nobs += 1


@dataclass
class FrozenSeasonalNaive:
    history: np.ndarray
    seasonal_period: int

    @classmethod
    def fit(
        cls,
        training_series,
        seasonal_period: int,
    ) -> "FrozenSeasonalNaive":
        y = np.asarray(
            training_series,
            dtype=float,
        ).reshape(-1)

        m = int(
            seasonal_period
        )

        if m <= 1:
            raise ValueError(
                "seasonal_naive is available only when m>1"
            )

        if y.size < m:
            raise ValueError(
                "insufficient training history "
                "for seasonal_naive"
            )

        if not np.all(np.isfinite(y)):
            raise ValueError(
                "seasonal_naive training series contains "
                "non-finite values"
            )

        return cls(
            history=y.copy(),
            seasonal_period=m,
        )

    def forecast_one(self) -> float:
        value = float(
            self.history[
                -self.seasonal_period
            ]
        )

        if not np.isfinite(value):
            raise ValueError(
                "seasonal_naive forecast is non-finite"
            )

        return value

    def observe(
        self,
        observed_value: float,
    ) -> None:
        observed = float(
            observed_value
        )

        if not np.isfinite(observed):
            raise ValueError(
                "observed value must be finite"
            )

        self.history = np.append(
            self.history,
            observed,
        )


@dataclass
class FrozenDrift:
    last_value: float
    slope: float
    nobs: int

    @classmethod
    def fit(
        cls,
        training_series,
    ) -> "FrozenDrift":
        y = np.asarray(
            training_series,
            dtype=float,
        ).reshape(-1)

        if y.size < 2:
            raise ValueError(
                "drift requires at least two "
                "training observations"
            )

        if not np.all(np.isfinite(y)):
            raise ValueError(
                "drift training series contains "
                "non-finite values"
            )

        slope = float(
            (y[-1] - y[0])
            / (y.size - 1)
        )

        if not np.isfinite(slope):
            raise ValueError(
                "drift slope is non-finite"
            )

        return cls(
            last_value=float(y[-1]),
            slope=slope,
            nobs=int(y.size),
        )

    def forecast_one(self) -> float:
        value = float(
            self.last_value
            + self.slope
        )

        if not np.isfinite(value):
            raise ValueError(
                "drift forecast is non-finite"
            )

        return value

    def observe(
        self,
        observed_value: float,
    ) -> None:
        observed = float(
            observed_value
        )

        if not np.isfinite(observed):
            raise ValueError(
                "observed value must be finite"
            )

        # B1: the drift parameter is frozen.
        # Only the observed level advances.
        self.last_value = observed
        self.nobs += 1


@dataclass(frozen=True)
class BaselineRunFailure:
    model: str
    stage: str
    origin: int | None
    error_type: str
    message: str


@dataclass
class BaselineRunResult:
    model: str
    status: str

    y_true: np.ndarray
    y_pred: np.ndarray
    test_indices: np.ndarray

    selected_config: dict
    selection_failures: list[CandidateFailure]

    failure: BaselineRunFailure | None

    update_policy: str
    substitution_used: bool = False

    @property
    def metrics_eligible(self) -> bool:
        return (
            self.status == "success"
            and self.failure is None
            and not self.substitution_used
        )


class BaselineQCError(RuntimeError):
    pass


V2_UPDATE_POLICY = (
    "fixed_model_parameters_with_observed_history"
)


def _frozen_lag_from_m(
    seasonal_period: int,
) -> int:
    m = int(
        seasonal_period
    )

    if m < 1:
        raise ValueError(
            "seasonal_period must be >= 1"
        )

    return int(
        min(
            12,
            max(
                5,
                m,
            ),
        )
    )


def _prepare_baseline_v2(
    training_series,
    model: str,
    seasonal_period: int,
):
    name = str(
        model
    ).strip().lower()

    m = int(
        seasonal_period
    )

    L = _frozen_lag_from_m(
        m
    )

    selection_failures = []

    if name == "naive_1":
        fitted = FrozenNaive1.fit(
            training_series
        )

        config = {
            "tuning": "none",
        }

    elif name == "seasonal_naive":
        fitted = FrozenSeasonalNaive.fit(
            training_series,
            seasonal_period=m,
        )

        config = {
            "tuning": "none",
            "seasonal_period": m,
        }

    elif name == "drift":
        fitted = FrozenDrift.fit(
            training_series
        )

        config = {
            "tuning": "none",
            "slope": fitted.slope,
        }

    elif name == "sarima":
        selection = select_sarima_aicc(
            training_series,
            seasonal_period=m,
        )

        fitted = selection.model

        selection_failures = list(
            selection.failures
        )

        config = {
            "order": selection.candidate.order,
            "seasonal_order": (
                selection.candidate.seasonal_order
            ),
            "trend": selection.candidate.trend,
            "aicc": float(
                selection.aicc
            ),
        }

    elif name == "ets":
        selection = select_ets_aicc(
            training_series,
            seasonal_period=m,
        )

        fitted = selection.model

        selection_failures = list(
            selection.failures
        )

        config = {
            "trend": selection.candidate.trend,
            "damped_trend": (
                selection.candidate.damped_trend
            ),
            "seasonal": (
                selection.candidate.seasonal
            ),
            "seasonal_periods": (
                selection.candidate.seasonal_periods
            ),
            "aicc": float(
                selection.aicc
            ),
        }

    elif name == "theta":
        fitted = FrozenTheta.fit(
            training_series,
            seasonal_period=m,
        )

        config = {
            "theta": 2.0,
            "seasonal_period": m,
        }

    elif name == "ridge_lag":
        selection = select_ridge(
            training_series,
            n_lags=L,
            seasonal_period=m,
            alphas=[
                1e-6,
                1e-4,
                1e-2,
                0.1,
                1.0,
                10.0,
                100.0,
            ],
        )

        fitted = FrozenLagRegressor.fit_ridge(
            training_series,
            n_lags=L,
            alpha=selection.alpha,
        )

        config = {
            "n_lags": L,
            "alpha": float(
                selection.alpha
            ),
            "validation_mase": float(
                selection.validation_mase
            ),
        }

    elif name == "svr_rbf":
        selection = select_svr(
            training_series,
            n_lags=L,
            seasonal_period=m,
            C_values=[
                0.1,
                1.0,
                10.0,
                100.0,
            ],
            epsilon_values=[
                0.01,
                0.05,
                0.1,
            ],
            gamma_values=[
                "scale",
                0.1,
                1.0,
            ],
        )

        fitted = FrozenLagRegressor.fit_svr(
            training_series,
            n_lags=L,
            C=selection.C,
            epsilon=selection.epsilon,
            gamma=selection.gamma,
        )

        config = {
            "n_lags": L,
            "C": float(
                selection.C
            ),
            "epsilon": float(
                selection.epsilon
            ),
            "gamma": selection.gamma,
            "validation_mase": float(
                selection.validation_mase
            ),
        }

    elif name == "xgboost":
        selection = select_xgboost(
            training_series,
            n_lags=L,
            seasonal_period=m,
            n_estimators_values=[
                100,
                300,
            ],
            max_depth_values=[
                2,
                3,
            ],
            learning_rate_values=[
                0.03,
                0.1,
            ],
            min_child_weight_values=[
                1,
                5,
            ],
        )

        fitted = FrozenLagRegressor.fit_xgboost(
            training_series,
            n_lags=L,
            n_estimators=(
                selection.n_estimators
            ),
            max_depth=(
                selection.max_depth
            ),
            learning_rate=(
                selection.learning_rate
            ),
            min_child_weight=(
                selection.min_child_weight
            ),
        )

        config = {
            "n_lags": L,
            "n_estimators": int(
                selection.n_estimators
            ),
            "max_depth": int(
                selection.max_depth
            ),
            "learning_rate": float(
                selection.learning_rate
            ),
            "min_child_weight": int(
                selection.min_child_weight
            ),
            "validation_mase": float(
                selection.validation_mase
            ),
        }

    else:
        raise ValueError(
            f"unsupported V2 baseline: {name}"
        )

    return (
        fitted,
        config,
        selection_failures,
    )


def run_baseline_v2(
    training_series,
    test_series,
    model: str,
    seasonal_period: int,
) -> BaselineRunResult:
    name = str(
        model
    ).strip().lower()

    train = np.asarray(
        training_series,
        dtype=float,
    ).reshape(-1)

    test = np.asarray(
        test_series,
        dtype=float,
    ).reshape(-1)

    if train.size < 1:
        raise ValueError(
            "training_series must not be empty"
        )

    if test.size < 1:
        raise ValueError(
            "test_series must not be empty"
        )

    if not np.all(
        np.isfinite(train)
    ):
        raise ValueError(
            "training_series contains non-finite values"
        )

    if not np.all(
        np.isfinite(test)
    ):
        raise ValueError(
            "test_series contains non-finite values"
        )

    indices = np.arange(
        train.size,
        train.size + test.size,
        dtype=int,
    )

    predictions = np.full(
        test.size,
        np.nan,
        dtype=float,
    )

    selected_config = {}
    selection_failures = []

    preparation_stage = (
        "selection"
        if name in {
            "sarima",
            "ets",
            "ridge_lag",
            "svr_rbf",
            "xgboost",
        }
        else "fit"
    )
    try:
        (
            fitted,
            selected_config,
            selection_failures,
        ) = _prepare_baseline_v2(
            train,
            model=name,
            seasonal_period=seasonal_period,
        )

    except Exception as exc:
        if isinstance(
            exc,
            BaselineSelectionError,
        ):
            selection_failures = list(
                exc.failures
            )

        result = BaselineRunResult(
            model=name,
            status="failed",
            y_true=test.copy(),
            y_pred=predictions,
            test_indices=indices,
            selected_config=selected_config,
            selection_failures=selection_failures,
            failure=BaselineRunFailure(
                model=name,
                stage=preparation_stage,
                origin=None,
                error_type=(
                    type(exc).__name__
                ),
                message=str(exc),
            ),
            update_policy=V2_UPDATE_POLICY,
            substitution_used=False,
        )

        assert_baseline_run_qc(
            result
        )

        return result

    failure = None

    for i, actual in enumerate(test):
        origin = int(
            indices[i]
        )

        try:
            prediction = float(
                fitted.forecast_one()
            )

            if not np.isfinite(
                prediction
            ):
                raise ValueError(
                    "forecast is non-finite"
                )

            predictions[i] = prediction

        except Exception as exc:
            failure = BaselineRunFailure(
                model=name,
                stage="forecast",
                origin=origin,
                error_type=(
                    type(exc).__name__
                ),
                message=str(exc),
            )

            break

        try:
            # Observed truth becomes available only
            # after forecasting this origin.
            fitted.observe(
                float(actual)
            )

        except Exception as exc:
            failure = BaselineRunFailure(
                model=name,
                stage="update",
                origin=origin,
                error_type=(
                    type(exc).__name__
                ),
                message=str(exc),
            )

            break

    status = (
        "success"
        if failure is None
        else "failed"
    )

    result = BaselineRunResult(
        model=name,
        status=status,
        y_true=test.copy(),
        y_pred=predictions,
        test_indices=indices,
        selected_config=selected_config,
        selection_failures=selection_failures,
        failure=failure,
        update_policy=V2_UPDATE_POLICY,
        substitution_used=False,
    )

    assert_baseline_run_qc(
        result
    )

    return result


def baseline_run_qc_violations(
    result: BaselineRunResult,
) -> tuple[str, ...]:
    violations = []

    n_true = int(
        np.asarray(
            result.y_true
        ).size
    )

    n_pred = int(
        np.asarray(
            result.y_pred
        ).size
    )

    n_indices = int(
        np.asarray(
            result.test_indices
        ).size
    )

    if not (
        n_true
        == n_pred
        == n_indices
    ):
        violations.append(
            "forecast length mismatch"
        )

    if result.status not in {
        "success",
        "failed",
    }:
        violations.append(
            "invalid run status"
        )

    if (
        result.update_policy
        != V2_UPDATE_POLICY
    ):
        violations.append(
            "invalid update policy provenance"
        )

    if result.substitution_used:
        violations.append(
            "forecast substitution is prohibited"
        )

    if (
        result.failure is not None
        and result.failure.stage not in {
            "fit",
            "selection",
            "forecast",
            "update",
        }
    ):
        violations.append(
            "invalid failure stage provenance"
        )

    if result.status == "success":
        if result.failure is not None:
            violations.append(
                "successful run contains failure provenance"
            )

        if not np.all(
            np.isfinite(
                result.y_pred
            )
        ):
            violations.append(
                "successful run contains non-finite prediction"
            )

        if not result.selected_config:
            violations.append(
                "successful run lacks selected-config provenance"
            )

    if result.status == "failed":
        if result.failure is None:
            violations.append(
                "failed run lacks failure provenance"
            )

    return tuple(
        violations
    )


def assert_baseline_run_qc(
    result: BaselineRunResult,
) -> None:
    violations = baseline_run_qc_violations(
        result
    )

    if violations:
        raise BaselineQCError(
            "; ".join(
                violations
            )
        )


def accuracy_arrays(
    result: BaselineRunResult,
) -> tuple[np.ndarray, np.ndarray]:
    assert_baseline_run_qc(
        result
    )

    if not result.metrics_eligible:
        raise BaselineQCError(
            "baseline run is not eligible "
            "for accuracy metrics"
        )

    return (
        np.asarray(
            result.y_true,
            dtype=float,
        ).copy(),
        np.asarray(
            result.y_pred,
            dtype=float,
        ).copy(),
    )
