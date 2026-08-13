from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np

SERIES_LENGTH = 180
TRAIN_FRACTION = 0.8
FORECAST_HORIZON = 1
CONFIRMATORY_NOISE_LEVELS = (0.05, 0.10, 0.20)
NOISE_FREE_LEVEL = 0.0

CUSTOM_GENERATORS = (
    "linear_ar",
    "nonlinear_sine",
    "regime_switching",
    "spiky_outbreak",
    "noisy_seasonal",
    "short_memory",
)
CANONICAL_GENERATORS = (
    "logistic_map",
    "henon_map",
    "lorenz63",
)
ALL_GENERATORS = CUSTOM_GENERATORS + CANONICAL_GENERATORS

_SEASONAL_PERIOD = {
    "linear_ar": 1,
    "nonlinear_sine": 24,
    "regime_switching": 1,
    "spiky_outbreak": 1,
    "noisy_seasonal": 12,
    "short_memory": 1,
    "logistic_map": 1,
    "henon_map": 1,
    "lorenz63": 1,
}

_NOISE_SEED_OFFSET = {
    "linear_ar": 1000,
    "nonlinear_sine": 2000,
    "regime_switching": 3000,
    "spiky_outbreak": 4000,
    "noisy_seasonal": 5000,
    "short_memory": 6000,
    "logistic_map": 7000,
    "henon_map": 8000,
    "lorenz63": 9000,
}

_NOISE_MINIMUM_SCALE = {
    "linear_ar": 1.0,
    "nonlinear_sine": 1.0,
    "regime_switching": 1.0,
    "spiky_outbreak": 5.0,
    "noisy_seasonal": 2.0,
    "short_memory": 1.0,
    "logistic_map": 1.0,
    "henon_map": 1.0,
    "lorenz63": 1.0,
}


@dataclass(frozen=True)
class SyntheticSeries:
    generator: str
    dgp_seed: int
    noise_level: float
    seasonal_period: int
    clean: np.ndarray
    observed: np.ndarray

    def __post_init__(self) -> None:
        if self.generator not in ALL_GENERATORS:
            raise ValueError(f"Unknown generator: {self.generator!r}")
        clean = np.asarray(self.clean, dtype=np.float64).reshape(-1).copy()
        observed = np.asarray(self.observed, dtype=np.float64).reshape(-1).copy()
        if clean.shape != (SERIES_LENGTH,) or observed.shape != (SERIES_LENGTH,):
            raise ValueError(f"Synthetic V2 series must have length {SERIES_LENGTH}.")
        if not np.all(np.isfinite(clean)) or not np.all(np.isfinite(observed)):
            raise ValueError("Synthetic series contains non-finite values.")
        clean.setflags(write=False)
        observed.setflags(write=False)
        object.__setattr__(self, "clean", clean)
        object.__setattr__(self, "observed", observed)


def list_generators() -> tuple[str, ...]:
    return ALL_GENERATORS


def seasonal_period(generator: str) -> int:
    _validate_generator(generator)
    return int(_SEASONAL_PERIOD[generator])


def _validate_generator(generator: str) -> None:
    if generator not in ALL_GENERATORS:
        raise ValueError(
            f"Unknown V2 synthetic generator {generator!r}; expected one of {ALL_GENERATORS}."
        )


def _validate_seed(dgp_seed: int) -> int:
    if isinstance(dgp_seed, (bool, np.bool_)) or not isinstance(dgp_seed, (int, np.integer)):
        raise TypeError("dgp_seed must be an integer.")
    return int(dgp_seed)


def _validate_noise_level(noise_level: float) -> float:
    value = float(noise_level)
    if not np.isfinite(value):
        raise ValueError("noise_level must be finite.")
    if value < 0.0:
        raise ValueError("noise_level must be nonnegative.")
    return value


def apply_relative_observation_noise(
    clean: np.ndarray,
    *,
    noise_level: float,
    noise_seed: int,
    minimum_scale: float,
) -> np.ndarray:
    clean_arr = np.asarray(clean, dtype=np.float64).reshape(-1)
    if len(clean_arr) == 0 or not np.all(np.isfinite(clean_arr)):
        raise ValueError("clean must be a non-empty finite vector.")
    level = _validate_noise_level(noise_level)
    min_scale = float(minimum_scale)
    if not np.isfinite(min_scale) or min_scale <= 0.0:
        raise ValueError("minimum_scale must be finite and positive.")
    if level == 0.0:
        return clean_arr.copy()

    scale = max(float(np.std(clean_arr, ddof=0)), min_scale)
    rng = np.random.default_rng(int(noise_seed))
    standard_normal = rng.normal(loc=0.0, scale=1.0, size=len(clean_arr))
    return clean_arr + level * scale * standard_normal


def _linear_ar_clean(dgp_seed: int) -> np.ndarray:
    rng = np.random.default_rng(dgp_seed)
    y = np.zeros(SERIES_LENGTH, dtype=np.float64)
    y[0] = rng.normal(loc=0.0, scale=1.0)
    for t in range(1, SERIES_LENGTH):
        y[t] = 0.70 * y[t - 1] + rng.normal(loc=0.0, scale=1.0)
    return y


def _nonlinear_sine_clean(_: int) -> np.ndarray:
    t = np.arange(SERIES_LENGTH, dtype=np.float64)
    s = np.sin(2.0 * np.pi * t / 24.0)
    return 10.0 * s + 3.5 * s * s + 0.03 * t


def _regime_switching_clean(dgp_seed: int) -> np.ndarray:
    rng = np.random.default_rng(dgp_seed)
    switch = int(round(SERIES_LENGTH * 0.55))
    y = np.zeros(SERIES_LENGTH, dtype=np.float64)
    y[0] = rng.normal(loc=0.0, scale=1.0)
    for t in range(1, SERIES_LENGTH):
        if t < switch:
            phi = 0.75
            innovation_sd = 0.8
            drift = 0.03
        else:
            phi = 0.25
            innovation_sd = 1.8
            drift = -0.01
        y[t] = drift * t + phi * y[t - 1] + rng.normal(loc=0.0, scale=innovation_sd)
    return y


def _spiky_outbreak_clean(dgp_seed: int) -> np.ndarray:
    rng = np.random.default_rng(dgp_seed)
    t = np.arange(SERIES_LENGTH, dtype=np.float64)
    y = 20.0 + 0.05 * t + rng.normal(loc=0.0, scale=1.0, size=SERIES_LENGTH)
    candidate_positions = np.arange(10, SERIES_LENGTH - 10, dtype=int)
    spike_positions = rng.choice(candidate_positions, size=5, replace=False)
    for pos in spike_positions:
        height = rng.uniform(20.0, 70.0)
        width = rng.uniform(1.5, 5.0)
        y += height * np.exp(-0.5 * ((t - float(pos)) / width) ** 2)
    return np.maximum(y, 0.0)


def _noisy_seasonal_clean(_: int) -> np.ndarray:
    t = np.arange(SERIES_LENGTH, dtype=np.float64)
    return (
        50.0
        + 0.04 * t
        + 8.0 * np.sin(2.0 * np.pi * t / 12.0)
        + 2.8 * np.cos(4.0 * np.pi * t / 12.0)
    )


def _short_memory_clean(dgp_seed: int) -> np.ndarray:
    rng = np.random.default_rng(dgp_seed)
    y = rng.normal(loc=0.0, scale=1.0, size=SERIES_LENGTH).astype(np.float64)
    for t in range(2, SERIES_LENGTH):
        y[t] = y[t] + 0.30 * y[t - 1] - 0.15 * y[t - 2]
    return y


def _logistic_map_clean(dgp_seed: int) -> np.ndarray:
    rng = np.random.default_rng(dgp_seed)
    x = np.float64(rng.uniform(0.1, 0.9))
    for _ in range(500):
        x = np.float64(4.0 * x * (1.0 - x))
    out = np.empty(SERIES_LENGTH, dtype=np.float64)
    out[0] = x
    for i in range(1, SERIES_LENGTH):
        x = np.float64(4.0 * x * (1.0 - x))
        out[i] = x
    return out


_HENON_BURN_IN_UPDATES = 1000
_HENON_FEASIBILITY_UPDATES = _HENON_BURN_IN_UPDATES + (SERIES_LENGTH - 1)
_HENON_ESCAPE_GUARD = 1_000_000.0
_HENON_MAX_CANDIDATE_PAIRS = 10_000


def _henon_update(
    x_old: np.float64,
    y_old: np.float64,
) -> tuple[np.float64, np.float64]:
    # Preserve the source-level arithmetic order used by the frozen V2 generator.
    x_new = np.float64(1.0 - 1.4 * x_old * x_old + y_old)
    y_new = np.float64(0.3 * x_old)
    return x_new, y_new


def _henon_candidate_is_feasible(
    x_initial: np.float64,
    y_initial: np.float64,
) -> bool:
    x = np.float64(x_initial)
    y = np.float64(y_initial)
    for _ in range(_HENON_FEASIBILITY_UPDATES):
        # Rejected candidates may escape explosively; suppress the expected floating-
        # point warnings and make the rejection decision from the explicit guard.
        with np.errstate(over="ignore", invalid="ignore"):
            x, y = _henon_update(x, y)
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
        if abs(float(x)) > _HENON_ESCAPE_GUARD or abs(float(y)) > _HENON_ESCAPE_GUARD:
            return False
    return True


def henon_initial_condition(dgp_seed: int) -> tuple[float, float, int]:
    """Return the first A008 basin-valid Hénon initial-condition pair.

    Candidate pairs are drawn sequentially from one ``default_rng(dgp_seed)``.
    The third returned value is the one-based accepted candidate number.
    """
    seed = _validate_seed(dgp_seed)
    rng = np.random.default_rng(seed)
    for candidate_number in range(1, _HENON_MAX_CANDIDATE_PAIRS + 1):
        x = np.float64(rng.uniform(-0.5, 0.5))
        y = np.float64(rng.uniform(-0.5, 0.5))
        if _henon_candidate_is_feasible(x, y):
            return float(x), float(y), candidate_number
    raise RuntimeError(
        "No basin-valid Hénon initial condition found within "
        f"{_HENON_MAX_CANDIDATE_PAIRS} candidate pairs for dgp_seed={seed}."
    )


def _henon_map_clean(dgp_seed: int) -> np.ndarray:
    x0, y0, _ = henon_initial_condition(dgp_seed)
    x = np.float64(x0)
    y = np.float64(y0)

    for _ in range(_HENON_BURN_IN_UPDATES):
        x, y = _henon_update(x, y)
    out = np.empty(SERIES_LENGTH, dtype=np.float64)
    out[0] = x
    for i in range(1, SERIES_LENGTH):
        x, y = _henon_update(x, y)
        out[i] = x
    return out


def _lorenz_rhs(state: np.ndarray) -> np.ndarray:
    x, y, z = state
    dx = 10.0 * (y - x)
    dy = x * (28.0 - z) - y
    dz = x * y - (8.0 / 3.0) * z
    return np.array([dx, dy, dz], dtype=np.float64)


def _lorenz_rk4_step(state: np.ndarray) -> np.ndarray:
    dt = 0.01
    k1 = _lorenz_rhs(state)
    k2 = _lorenz_rhs(state + 0.5 * dt * k1)
    k3 = _lorenz_rhs(state + 0.5 * dt * k2)
    k4 = _lorenz_rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _lorenz63_clean(dgp_seed: int) -> np.ndarray:
    rng = np.random.default_rng(dgp_seed)
    state = np.array(
        [
            rng.uniform(-15.0, 15.0),
            rng.uniform(-20.0, 20.0),
            rng.uniform(5.0, 35.0),
        ],
        dtype=np.float64,
    )
    for _ in range(5000):
        state = _lorenz_rk4_step(state)
    out = np.empty(SERIES_LENGTH, dtype=np.float64)
    out[0] = state[0]
    for i in range(1, SERIES_LENGTH):
        for _ in range(10):
            state = _lorenz_rk4_step(state)
        out[i] = state[0]
    return out


_CLEAN_GENERATORS = {
    "linear_ar": _linear_ar_clean,
    "nonlinear_sine": _nonlinear_sine_clean,
    "regime_switching": _regime_switching_clean,
    "spiky_outbreak": _spiky_outbreak_clean,
    "noisy_seasonal": _noisy_seasonal_clean,
    "short_memory": _short_memory_clean,
    "logistic_map": _logistic_map_clean,
    "henon_map": _henon_map_clean,
    "lorenz63": _lorenz63_clean,
}


@lru_cache(maxsize=512)
def _generate_clean_cached(generator: str, dgp_seed: int) -> np.ndarray:
    _validate_generator(generator)
    clean = np.asarray(_CLEAN_GENERATORS[generator](dgp_seed), dtype=np.float64).reshape(-1)
    if clean.shape != (SERIES_LENGTH,):
        raise RuntimeError(f"Generator {generator!r} returned an invalid length.")
    if not np.all(np.isfinite(clean)):
        raise RuntimeError(f"Generator {generator!r} returned non-finite values.")
    clean = clean.copy()
    clean.setflags(write=False)
    return clean


def generate_clean_series(generator: str, *, dgp_seed: int) -> np.ndarray:
    _validate_generator(generator)
    seed = _validate_seed(dgp_seed)
    return _generate_clean_cached(generator, seed).copy()


def generate_synthetic_series(
    generator: str,
    *,
    dgp_seed: int,
    noise_level: float,
) -> SyntheticSeries:
    _validate_generator(generator)
    seed = _validate_seed(dgp_seed)
    level = _validate_noise_level(noise_level)
    clean = generate_clean_series(generator, dgp_seed=seed)
    observed = apply_relative_observation_noise(
        clean,
        noise_level=level,
        noise_seed=seed + _NOISE_SEED_OFFSET[generator],
        minimum_scale=_NOISE_MINIMUM_SCALE[generator],
    )
    if generator == "spiky_outbreak":
        observed = np.maximum(observed, 0.0)
    return SyntheticSeries(
        generator=generator,
        dgp_seed=seed,
        noise_level=level,
        seasonal_period=seasonal_period(generator),
        clean=clean,
        observed=observed,
    )


def generate_noise_bundle(
    generator: str,
    *,
    dgp_seed: int,
    noise_levels: Iterable[float] = CONFIRMATORY_NOISE_LEVELS,
) -> dict[float, SyntheticSeries]:
    _validate_generator(generator)
    seed = _validate_seed(dgp_seed)
    levels = tuple(_validate_noise_level(v) for v in noise_levels)
    if len(levels) == 0:
        raise ValueError("noise_levels must be non-empty.")
    if len(set(levels)) != len(levels):
        raise ValueError("noise_levels must not contain duplicates.")
    clean = generate_clean_series(generator, dgp_seed=seed)
    result: dict[float, SyntheticSeries] = {}
    for level in levels:
        observed = apply_relative_observation_noise(
            clean,
            noise_level=level,
            noise_seed=seed + _NOISE_SEED_OFFSET[generator],
            minimum_scale=_NOISE_MINIMUM_SCALE[generator],
        )
        if generator == "spiky_outbreak":
            observed = np.maximum(observed, 0.0)
        result[level] = SyntheticSeries(
            generator=generator,
            dgp_seed=seed,
            noise_level=level,
            seasonal_period=seasonal_period(generator),
            clean=clean,
            observed=observed,
        )
    return result


__all__ = [
    "SERIES_LENGTH",
    "TRAIN_FRACTION",
    "FORECAST_HORIZON",
    "CONFIRMATORY_NOISE_LEVELS",
    "NOISE_FREE_LEVEL",
    "CUSTOM_GENERATORS",
    "CANONICAL_GENERATORS",
    "ALL_GENERATORS",
    "SyntheticSeries",
    "list_generators",
    "seasonal_period",
    "apply_relative_observation_noise",
    "henon_initial_condition",
    "generate_clean_series",
    "generate_synthetic_series",
    "generate_noise_bundle",
]
