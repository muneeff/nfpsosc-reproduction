from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import TSKModel


SC_SQUASH_FACTOR = 1.25
SC_ACCEPT_RATIO = 0.5
SC_REJECT_RATIO = 0.15
SC_WIDTH_DENOMINATOR = np.sqrt(8.0)
SCALER_OUT_MIN = 0.1
SCALER_OUT_MAX = 0.9
FEATURE_RANGE_FLOOR = 1e-3


def _finite_1d(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if len(arr) < 2:
        raise ValueError(f"{name} must contain at least two observations.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _finite_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X2 = np.asarray(X, dtype=float)
    y1 = np.asarray(y, dtype=float).reshape(-1)
    if X2.ndim != 2 or len(X2) < 1:
        raise ValueError("X_fit must be a non-empty 2D feature matrix.")
    if len(X2) != len(y1):
        raise ValueError("X_fit and y_fit must contain the same number of samples.")
    if not np.all(np.isfinite(X2)) or not np.all(np.isfinite(y1)):
        raise ValueError("X_fit and y_fit must contain only finite values.")
    return X2, y1


@dataclass(frozen=True)
class FrozenMinMaxScaler1D:
    """Frozen V2 affine scaler fitted on fitting observations only.

    The output interval [0.1, 0.9] is fixed by V2-A004. Values outside the
    fitting range are deliberately extrapolated rather than clipped so that
    validation/test observations cannot alter the fitted transform.
    """

    data_min: float
    data_max: float
    out_min: float = SCALER_OUT_MIN
    out_max: float = SCALER_OUT_MAX

    def __post_init__(self) -> None:
        vals = (self.data_min, self.data_max, self.out_min, self.out_max)
        if not all(np.isfinite(v) for v in vals):
            raise ValueError("Scaler parameters must be finite.")
        if not self.data_max > self.data_min:
            raise ValueError("Fitting observations must have strictly positive range.")
        if not self.out_max > self.out_min:
            raise ValueError("Scaler output maximum must exceed output minimum.")
        if (self.out_min, self.out_max) != (SCALER_OUT_MIN, SCALER_OUT_MAX):
            raise ValueError("V2 scaler output interval is frozen at [0.1, 0.9].")

    @classmethod
    def fit(cls, fitting_observations: np.ndarray) -> "FrozenMinMaxScaler1D":
        raw = _finite_1d(fitting_observations, name="fitting_observations")
        return cls(data_min=float(np.min(raw)), data_max=float(np.max(raw)))

    @property
    def data_range(self) -> float:
        return float(self.data_max - self.data_min)

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Values to transform must be finite.")
        scale = (self.out_max - self.out_min) / self.data_range
        return self.out_min + (arr - self.data_min) * scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Values to inverse-transform must be finite.")
        scale = self.data_range / (self.out_max - self.out_min)
        return self.data_min + (arr - self.out_min) * scale


@dataclass(frozen=True)
class SubtractiveClusteringResult:
    centers: np.ndarray
    centers_normalized: np.ndarray
    potentials: np.ndarray
    data_min: np.ndarray
    data_range: np.ndarray
    radius: float


@dataclass(frozen=True)
class AntecedentBounds:
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float).reshape(-1)
        upper = np.asarray(self.upper, dtype=float).reshape(-1)
        if len(lower) < 1 or lower.shape != upper.shape:
            raise ValueError("Antecedent bounds must be non-empty and shape-matched.")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("Antecedent bounds must be finite.")
        if np.any(upper <= lower):
            raise ValueError("Every antecedent upper bound must exceed its lower bound.")
        object.__setattr__(self, "lower", lower.copy())
        object.__setattr__(self, "upper", upper.copy())


@dataclass(frozen=True)
class TSKInitialization:
    model: TSKModel
    clustering: SubtractiveClusteringResult
    bounds: AntecedentBounds
    particle0_unclipped: np.ndarray
    particle0: np.ndarray


def _normalize_unit(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = np.min(data, axis=0)
    hi = np.max(data, axis=0)
    span = hi - lo
    safe_span = np.where(span > 0.0, span, 1.0)
    normalized = (data - lo) / safe_span
    return normalized, lo, safe_span


def subtractive_clustering(
    data: np.ndarray,
    *,
    radius: float,
) -> SubtractiveClusteringResult:
    """Deterministic Chiu-style subtractive clustering inherited for V2-A004."""
    Z = np.asarray(data, dtype=float)
    if Z.ndim != 2 or len(Z) == 0:
        raise ValueError("data must be a non-empty 2D array.")
    if not np.all(np.isfinite(Z)):
        raise ValueError("data must contain only finite values.")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be a finite positive scalar.")

    Zn, lo, span = _normalize_unit(Z)
    diff = Zn[:, None, :] - Zn[None, :, :]
    squared_distance = np.sum(diff * diff, axis=2)
    alpha = 4.0 / (radius * radius)
    potentials = np.sum(np.exp(-alpha * squared_distance), axis=1)
    working = potentials.copy()

    centers_idx: list[int] = []
    center_potentials: list[float] = []
    first_potential: float | None = None
    rb = SC_SQUASH_FACTOR * radius
    beta = 4.0 / (rb * rb)

    while True:
        candidate_idx = int(np.argmax(working))
        candidate_potential = float(working[candidate_idx])
        if candidate_potential <= 0.0:
            break

        if first_potential is None:
            accept = True
            first_potential = candidate_potential
        elif candidate_potential >= SC_ACCEPT_RATIO * first_potential:
            accept = True
        elif candidate_potential <= SC_REJECT_RATIO * first_potential:
            break
        else:
            candidate = Zn[candidate_idx]
            chosen = Zn[np.asarray(centers_idx, dtype=int)]
            min_distance = float(
                np.min(np.linalg.norm(chosen - candidate, axis=1))
            )
            accept = (
                min_distance / radius
                + candidate_potential / first_potential
            ) >= 1.0

        if not accept:
            working[candidate_idx] = 0.0
            continue

        centers_idx.append(candidate_idx)
        center_potentials.append(candidate_potential)
        center = Zn[candidate_idx]
        distance_to_center = np.sum((Zn - center) ** 2, axis=1)
        working = np.maximum(
            0.0,
            working
            - candidate_potential * np.exp(-beta * distance_to_center),
        )

    if not centers_idx:
        raise RuntimeError("Subtractive clustering failed to select a center.")

    idx = np.asarray(centers_idx, dtype=int)
    centers_normalized = Zn[idx]
    centers = centers_normalized * span + lo
    return SubtractiveClusteringResult(
        centers=centers.copy(),
        centers_normalized=centers_normalized.copy(),
        potentials=np.asarray(center_potentials, dtype=float),
        data_min=lo.copy(),
        data_range=span.copy(),
        radius=float(radius),
    )


def feature_ranges(X_fit: np.ndarray) -> np.ndarray:
    X = np.asarray(X_fit, dtype=float)
    if X.ndim != 2 or len(X) < 1 or not np.all(np.isfinite(X)):
        raise ValueError("X_fit must be a non-empty finite 2D matrix.")
    return np.maximum(np.ptp(X, axis=0), FEATURE_RANGE_FLOOR)


def make_antecedent_bounds(X_fit: np.ndarray, *, n_rules: int) -> AntecedentBounds:
    X = np.asarray(X_fit, dtype=float)
    if X.ndim != 2 or len(X) < 1 or not np.all(np.isfinite(X)):
        raise ValueError("X_fit must be a non-empty finite 2D matrix.")
    if not isinstance(n_rules, (int, np.integer)) or int(n_rules) < 1:
        raise ValueError("n_rules must be a positive integer.")

    ranges = feature_ranges(X)
    feature_min = np.min(X, axis=0)
    feature_max = np.max(X, axis=0)
    center_lower = np.tile(feature_min - 0.05 * ranges, int(n_rules))
    center_upper = np.tile(feature_max + 0.05 * ranges, int(n_rules))
    sigma_lower = np.tile(0.05 * ranges, int(n_rules))
    sigma_upper = np.tile(1.00 * ranges, int(n_rules))
    return AntecedentBounds(
        lower=np.concatenate([center_lower, sigma_lower]),
        upper=np.concatenate([center_upper, sigma_upper]),
    )


def initialize_tsk_from_subtractive_clustering(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    *,
    radius: float,
) -> TSKInitialization:
    """Create V2 particle-0 TSK antecedents from fitting data only.

    ``X_fit`` and ``y_fit`` must already be in the frozen scaled domain.
    Consequents are intentionally zero here; the objective layer fits them with
    an explicitly supplied frozen ridge alpha for each antecedent candidate.
    """
    X, y = _finite_xy(X_fit, y_fit)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be a finite positive scalar.")

    joint = np.column_stack([X, y])
    clustering = subtractive_clustering(joint, radius=float(radius))
    centers = clustering.centers[:, : X.shape[1]].copy()
    ranges = feature_ranges(X)
    sigma_per_feature = float(radius) * ranges / SC_WIDTH_DENOMINATOR
    sigmas = np.tile(sigma_per_feature, (len(centers), 1))

    model = TSKModel(
        centers=centers,
        sigmas=sigmas,
        consequents=np.zeros(
            (len(centers), X.shape[1] + 1),
            dtype=float,
        ),
    )
    bounds = make_antecedent_bounds(X, n_rules=model.n_rules)
    particle0_unclipped = model.antecedent_vector()
    particle0 = np.clip(
        particle0_unclipped,
        bounds.lower,
        bounds.upper,
    )

    return TSKInitialization(
        model=model,
        clustering=clustering,
        bounds=bounds,
        particle0_unclipped=particle0_unclipped.copy(),
        particle0=particle0.copy(),
    )
