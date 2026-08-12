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