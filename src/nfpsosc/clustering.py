from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubtractiveClusteringResult:
    centers: np.ndarray
    centers_normalized: np.ndarray
    potentials: np.ndarray
    data_min: np.ndarray
    data_range: np.ndarray
    radius: float


def _normalize_unit(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = np.min(data, axis=0)
    hi = np.max(data, axis=0)
    span = hi - lo
    safe_span = np.where(span > 0.0, span, 1.0)
    normalized = (data - lo) / safe_span
    return normalized, lo, safe_span


def subtractive_clustering(
    data: np.ndarray,
    radius: float = 0.55,
    squash_factor: float = 1.25,
    accept_ratio: float = 0.5,
    reject_ratio: float = 0.15,
    max_clusters: int | None = None,
) -> SubtractiveClusteringResult:
    """Subtractive clustering following the standard Chiu-style procedure.

    The implementation is transparent and deterministic. It is designed to
    reproduce the behavior intended by MATLAB genfis2/subclust, but it is not
    claimed to be bit-for-bit identical to a specific MATLAB release.
    """
    Z = np.asarray(data, dtype=float)
    if Z.ndim != 2 or len(Z) == 0:
        raise ValueError("data must be a non-empty 2D array.")
    if not np.all(np.isfinite(Z)):
        raise ValueError("data contains non-finite values.")
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if not 0.0 < reject_ratio < accept_ratio < 1.0:
        raise ValueError("Require 0 < reject_ratio < accept_ratio < 1.")

    Zn, lo, span = _normalize_unit(Z)
    diff = Zn[:, None, :] - Zn[None, :, :]
    squared_distance = np.sum(diff * diff, axis=2)
    alpha = 4.0 / (radius * radius)
    potentials = np.sum(np.exp(-alpha * squared_distance), axis=1)
    working = potentials.copy()

    centers_idx: list[int] = []
    center_potentials: list[float] = []
    first_potential: float | None = None
    rb = squash_factor * radius
    beta = 4.0 / (rb * rb)

    while True:
        candidate_idx = int(np.argmax(working))
        candidate_potential = float(working[candidate_idx])
        if candidate_potential <= 0.0:
            break

        if first_potential is None:
            accept = True
            first_potential = candidate_potential
        elif candidate_potential >= accept_ratio * first_potential:
            accept = True
        elif candidate_potential <= reject_ratio * first_potential:
            break
        else:
            candidate = Zn[candidate_idx]
            chosen = Zn[np.array(centers_idx)]
            min_distance = float(np.min(np.linalg.norm(chosen - candidate, axis=1)))
            accept = (min_distance / radius + candidate_potential / first_potential) >= 1.0

        if not accept:
            working[candidate_idx] = 0.0
            continue

        centers_idx.append(candidate_idx)
        center_potentials.append(candidate_potential)
        center = Zn[candidate_idx]
        distance_to_center = np.sum((Zn - center) ** 2, axis=1)
        working = np.maximum(
            0.0,
            working - candidate_potential * np.exp(-beta * distance_to_center),
        )

        if max_clusters is not None and len(centers_idx) >= max_clusters:
            break

    if not centers_idx:
        raise RuntimeError("Subtractive clustering failed to select a center.")

    centers_normalized = Zn[np.array(centers_idx)]
    centers = centers_normalized * span + lo
    return SubtractiveClusteringResult(
        centers=centers,
        centers_normalized=centers_normalized,
        potentials=np.asarray(center_potentials),
        data_min=lo,
        data_range=span,
        radius=radius,
    )


def initialize_tsk_from_sc(
    X: np.ndarray,
    y: np.ndarray,
    radius: float = 0.55,
    ridge: float = 1e-6,
):
    """Create a first-order TSK model from joint input-output SC centers."""
    from .fis import TSKFIS

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be 2D and match y.")

    joint = np.column_stack([X, y])
    result = subtractive_clustering(joint, radius=radius)
    centers = result.centers[:, : X.shape[1]]

    feature_range = np.ptp(X, axis=0)
    feature_range = np.where(feature_range > 0.0, feature_range, 1.0)
    # Standard Gaussian width used by subtractive-clustering FIS construction.
    sigmas_per_feature = radius * feature_range / np.sqrt(8.0)
    sigmas = np.tile(sigmas_per_feature, (len(centers), 1))

    model = TSKFIS(
        centers=centers,
        sigmas=sigmas,
        consequents=np.zeros((len(centers), X.shape[1] + 1), dtype=float),
    )
    model.fit_consequents(X, y, ridge=ridge)
    return model, result
