from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MinMaxScaler:
    """Min-max scaler with an explicit output interval.

    The thesis maps the full revenue range to [0.1, 0.9]. For research-grade
    evaluation, fit the scaler on the training portion only to avoid leakage.
    """

    data_min: float
    data_max: float
    out_min: float = 0.1
    out_max: float = 0.9

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        out_min: float = 0.1,
        out_max: float = 0.9,
    ) -> "MinMaxScaler":
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError("Scaler requires finite, non-empty values.")
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi <= lo:
            raise ValueError("Scaler requires data_max > data_min.")
        return cls(lo, hi, out_min, out_max)

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        scale = (self.out_max - self.out_min) / (self.data_max - self.data_min)
        return self.out_min + (arr - self.data_min) * scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        scale = (self.data_max - self.data_min) / (self.out_max - self.out_min)
        return self.data_min + (arr - self.out_min) * scale


def load_revenue_series(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the monthly Yemen revenue series."""
    path = Path(csv_path)
    df = pd.read_csv(path)
    required = {"date", "revenue_million_yer"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["revenue_million_yer"] = pd.to_numeric(
        df["revenue_million_yer"], errors="raise"
    )
    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found.")
    if not np.all(np.isfinite(df["revenue_million_yer"].to_numpy())):
        raise ValueError("Revenue column contains non-finite values.")
    return df


def create_lagged_data(
    series: np.ndarray,
    n_lags: int = 5,
    order: Literal["chronological", "matlab_delay"] = "chronological",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a univariate series into one-step-ahead supervised samples.

    chronological:
        [y(t-n_lags), ..., y(t-1)] -> y(t), matching the thesis appendix table.
    matlab_delay:
        [y(t-1), ..., y(t-n_lags)] -> y(t), matching CreateTimeSeriesData.m
        when Delays = 1:n_lags.
    """
    y = np.asarray(series, dtype=float).reshape(-1)
    if n_lags < 1 or len(y) <= n_lags:
        raise ValueError("n_lags must be positive and smaller than series length.")

    X = np.array([y[i - n_lags : i] for i in range(n_lags, len(y))])
    if order == "matlab_delay":
        X = X[:, ::-1]
    elif order != "chronological":
        raise ValueError(f"Unsupported lag order: {order}")
    target = y[n_lags:]
    return X, target


def matlab_round_positive(value: float) -> int:
    """MATLAB-style rounding for non-negative values (0.5 rounds upward)."""
    if value < 0:
        raise ValueError("Only non-negative values are supported here.")
    return int(np.floor(value + 0.5))


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    train_fraction: float = 0.85,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same sample count.")
    n_train = matlab_round_positive(train_fraction * len(X))
    if n_train <= 0 or n_train >= len(X):
        raise ValueError("Split leaves an empty training or test set.")
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]
