"""
Synthetic time-series generators for the SS-NFPSO Q1 experiment pipeline.

These generators are intentionally simple, transparent, and deterministic under
a random seed. They are used to stress-test stability, sensitivity, sparsity,
and generalization claims without relying only on private or domain-specific
real data.

All public experiment scripts should treat these series as controlled test
processes, not as evidence of real-world forecasting superiority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticSeriesSpec:
    """Specification for one generated synthetic time series."""

    name: str
    length: int = 180
    noise_level: float = 0.05
    seed: int = 1
    season_length: int = 12


def _rng(seed: int) -> np.random.Generator:
    """Return a NumPy random generator with a fixed seed."""

    return np.random.default_rng(int(seed))


def _relative_noise(
    y: np.ndarray,
    noise_level: float,
    seed: int,
    minimum_scale: float = 1.0,
) -> np.ndarray:
    """
    Add Gaussian noise scaled relative to the standard deviation of the signal.

    Parameters
    ----------
    y:
        Base signal.
    noise_level:
        Relative noise level. For example, 0.10 means 10% of the signal
        standard deviation.
    seed:
        Random seed.
    minimum_scale:
        Lower bound for the noise scale to avoid zero-noise degenerate cases
        when the base signal is nearly constant.
    """

    y = np.asarray(y, dtype=float)
    if noise_level <= 0:
        return y.copy()

    scale = max(float(np.nanstd(y)), float(minimum_scale))
    noise = _rng(seed).normal(loc=0.0, scale=float(noise_level) * scale, size=len(y))
    return y + noise


def _to_frame(name: str, y: np.ndarray) -> pd.DataFrame:
    """Convert a generated array into a standard DataFrame."""

    y = np.asarray(y, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.RangeIndex(start=0, stop=len(y), step=1),
            "value": y,
            "series_name": name,
        }
    )


def generate_linear_ar(
    length: int = 180,
    noise_level: float = 0.05,
    seed: int = 1,
    phi: float = 0.70,
    intercept: float = 0.0,
) -> pd.DataFrame:
    """
    Generate a short linear autoregressive process.

    This process should be easy for simple statistical baselines. It is useful
    for checking whether the proposed NFPSO formulation overfits when the true
    data-generating mechanism is simple.
    """

    if length < 10:
        raise ValueError("length must be at least 10")

    gen = _rng(seed)
    y = np.zeros(length, dtype=float)
    y[0] = gen.normal()

    innovation_scale = 1.0
    for t in range(1, length):
        y[t] = intercept + phi * y[t - 1] + gen.normal(scale=innovation_scale)

    y = _relative_noise(y, noise_level=noise_level, seed=seed + 1000)
    return _to_frame("synthetic_linear_ar", y)


def generate_nonlinear_sine(
    length: int = 180,
    noise_level: float = 0.05,
    seed: int = 1,
    period: int = 24,
    amplitude: float = 10.0,
    nonlinear_strength: float = 0.35,
) -> pd.DataFrame:
    """
    Generate a smooth nonlinear sinusoidal process.

    This process is useful for testing whether nonlinear models gain an
    advantage while remaining stable under controlled noise.
    """

    if length < 10:
        raise ValueError("length must be at least 10")
    if period <= 1:
        raise ValueError("period must be greater than 1")

    t = np.arange(length, dtype=float)
    base = amplitude * np.sin(2.0 * np.pi * t / period)
    nonlinear = nonlinear_strength * amplitude * np.sin(2.0 * np.pi * t / period) ** 2
    slow_trend = 0.03 * t
    y = base + nonlinear + slow_trend

    y = _relative_noise(y, noise_level=noise_level, seed=seed + 2000)
    return _to_frame("synthetic_nonlinear_sine", y)


def generate_regime_switching(
    length: int = 180,
    noise_level: float = 0.05,
    seed: int = 1,
    switch_fraction: float = 0.55,
) -> pd.DataFrame:
    """
    Generate a process with a structural regime change.

    The first regime is smoother and more autoregressive. The second regime has
    different persistence and higher volatility. This tests failure modes under
    changing temporal structure.
    """

    if length < 20:
        raise ValueError("length must be at least 20")
    if not 0.1 < switch_fraction < 0.9:
        raise ValueError("switch_fraction must be between 0.1 and 0.9")

    gen = _rng(seed)
    switch = int(round(length * switch_fraction))
    y = np.zeros(length, dtype=float)
    y[0] = gen.normal()

    for t in range(1, length):
        if t < switch:
            phi = 0.75
            scale = 0.8
            drift = 0.03
        else:
            phi = 0.25
            scale = 1.8
            drift = -0.01
        y[t] = drift * t + phi * y[t - 1] + gen.normal(scale=scale)

    y = _relative_noise(y, noise_level=noise_level, seed=seed + 3000)
    return _to_frame("synthetic_regime_switching", y)


def generate_spiky_outbreak(
    length: int = 180,
    noise_level: float = 0.05,
    seed: int = 1,
    baseline: float = 20.0,
    n_spikes: int = 5,
) -> pd.DataFrame:
    """
    Generate an outbreak-like positive series with abrupt spikes.

    This process is useful for testing boundedness, extrapolation behavior, and
    residual/failure-case analysis.
    """

    if length < 30:
        raise ValueError("length must be at least 30")
    if n_spikes < 1:
        raise ValueError("n_spikes must be at least 1")

    gen = _rng(seed)
    t = np.arange(length, dtype=float)

    y = baseline + 0.05 * t + gen.normal(scale=1.0, size=length)
    candidate_positions = np.arange(10, length - 10)
    spike_positions = gen.choice(candidate_positions, size=min(n_spikes, len(candidate_positions)), replace=False)

    for pos in spike_positions:
        height = gen.uniform(20.0, 70.0)
        width = gen.uniform(1.5, 5.0)
        y += height * np.exp(-0.5 * ((t - pos) / width) ** 2)

    y = np.maximum(y, 0.0)
    y = _relative_noise(y, noise_level=noise_level, seed=seed + 4000, minimum_scale=5.0)
    y = np.maximum(y, 0.0)
    return _to_frame("synthetic_spiky_outbreak", y)


def generate_noisy_seasonal(
    length: int = 180,
    noise_level: float = 0.05,
    seed: int = 1,
    season_length: int = 12,
    amplitude: float = 8.0,
    trend: float = 0.04,
) -> pd.DataFrame:
    """
    Generate a seasonal process with trend and noise.

    This process is useful for comparing NFPSO against seasonal naive, ETS,
    Theta, and ARIMA-type baselines.
    """

    if length < season_length * 3:
        raise ValueError("length should cover at least three seasonal cycles")
    if season_length <= 1:
        raise ValueError("season_length must be greater than 1")

    t = np.arange(length, dtype=float)
    seasonal = amplitude * np.sin(2.0 * np.pi * t / season_length)
    harmonic = 0.35 * amplitude * np.cos(4.0 * np.pi * t / season_length)
    y = 50.0 + trend * t + seasonal + harmonic

    y = _relative_noise(y, noise_level=noise_level, seed=seed + 5000, minimum_scale=2.0)
    return _to_frame("synthetic_noisy_seasonal", y)


def generate_short_memory_process(
    length: int = 180,
    noise_level: float = 0.05,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Generate a weakly dependent short-memory process.

    This is a diagnostic series where complex models should not automatically
    be expected to outperform simple baselines.
    """

    if length < 10:
        raise ValueError("length must be at least 10")

    gen = _rng(seed)
    y = gen.normal(size=length)
    for t in range(2, length):
        y[t] += 0.30 * y[t - 1] - 0.15 * y[t - 2]

    y = _relative_noise(y, noise_level=noise_level, seed=seed + 6000)
    return _to_frame("synthetic_short_memory_process", y)


_GENERATORS: Dict[str, Callable[..., pd.DataFrame]] = {
    "linear_ar": generate_linear_ar,
    "synthetic_linear_ar": generate_linear_ar,
    "nonlinear_sine": generate_nonlinear_sine,
    "synthetic_nonlinear_sine": generate_nonlinear_sine,
    "regime_switching": generate_regime_switching,
    "synthetic_regime_switching": generate_regime_switching,
    "spiky_outbreak": generate_spiky_outbreak,
    "synthetic_spiky_outbreak": generate_spiky_outbreak,
    "noisy_seasonal": generate_noisy_seasonal,
    "synthetic_noisy_seasonal": generate_noisy_seasonal,
    "short_memory": generate_short_memory_process,
    "synthetic_short_memory_process": generate_short_memory_process,
}


def list_generators() -> list[str]:
    """Return available generator names."""

    return sorted(_GENERATORS.keys())


def make_synthetic_series(
    name: str,
    length: int = 180,
    noise_level: float = 0.05,
    seed: int = 1,
    season_length: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Generate a named synthetic series.

    Parameters
    ----------
    name:
        Generator name. See `list_generators()`.
    length:
        Number of observations.
    noise_level:
        Relative noise level.
    seed:
        Random seed.
    season_length:
        Optional seasonal period, used by seasonal generators.
    kwargs:
        Extra generator-specific arguments.
    """

    if name not in _GENERATORS:
        available = ", ".join(list_generators())
        raise ValueError(f"Unknown synthetic generator '{name}'. Available: {available}")

    generator = _GENERATORS[name]
    if season_length is not None:
        kwargs["season_length"] = season_length

    df = generator(length=length, noise_level=noise_level, seed=seed, **kwargs)
    df["generator"] = name
    df["length"] = int(length)
    df["noise_level"] = float(noise_level)
    df["seed"] = int(seed)
    return df


def make_synthetic_panel(
    generators: list[str],
    lengths: list[int],
    noise_levels: list[float],
    seeds: list[int],
    season_length: int = 12,
) -> pd.DataFrame:
    """
    Generate a long-format panel of synthetic series.

    The output contains the columns:
    date, value, series_name, generator, length, noise_level, seed, panel_id.
    """

    frames: list[pd.DataFrame] = []
    for generator in generators:
        for length in lengths:
            for noise_level in noise_levels:
                for seed in seeds:
                    kwargs = {}
                    if "seasonal" in generator:
                        kwargs["season_length"] = season_length
                    df = make_synthetic_series(
                        generator,
                        length=length,
                        noise_level=noise_level,
                        seed=seed,
                        **kwargs,
                    )
                    df["panel_id"] = f"{generator}_n{length}_noise{noise_level}_seed{seed}"
                    frames.append(df)

    if not frames:
        raise ValueError("No synthetic series were generated. Check input lists.")

    return pd.concat(frames, ignore_index=True)


def save_synthetic_series(
    path: str,
    name: str,
    length: int = 180,
    noise_level: float = 0.05,
    seed: int = 1,
    **kwargs,
) -> None:
    """Generate one synthetic series and save it as CSV."""

    df = make_synthetic_series(
        name=name,
        length=length,
        noise_level=noise_level,
        seed=seed,
        **kwargs,
    )
    df.to_csv(path, index=False)


__all__ = [
    "SyntheticSeriesSpec",
    "generate_linear_ar",
    "generate_nonlinear_sine",
    "generate_regime_switching",
    "generate_spiky_outbreak",
    "generate_noisy_seasonal",
    "generate_short_memory_process",
    "list_generators",
    "make_synthetic_series",
    "make_synthetic_panel",
    "save_synthetic_series",
]
