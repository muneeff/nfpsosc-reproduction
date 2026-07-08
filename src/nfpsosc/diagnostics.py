from __future__ import annotations

import numpy as np

from .fis import TSKFIS


def model_diagnostics(
    model: TSKFIS,
    X: np.ndarray,
    perturbation_scale: float = 0.01,
    perturbations_per_sample: int = 8,
    seed: int = 123,
) -> dict[str, float]:
    X = np.asarray(X, dtype=float)
    pred = model.predict(X)
    jac = model.jacobian(X)
    jac_norm = np.linalg.norm(jac, axis=1)
    _, activation_sum = model.normalized_weights(X)

    rng = np.random.default_rng(seed)
    ratios: list[float] = []
    feature_scale = np.std(X, axis=0, ddof=1)
    feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
    for _ in range(perturbations_per_sample):
        delta = rng.normal(size=X.shape) * feature_scale * perturbation_scale
        perturbed = model.predict(X + delta)
        numerator = np.abs(perturbed - pred)
        denominator = np.linalg.norm(delta, axis=1)
        ratios.extend((numerator / np.maximum(denominator, 1e-12)).tolist())

    vector = model.to_vector()
    return {
        "n_rules": float(model.n_rules),
        "n_parameters": float(len(vector)),
        "min_activation_sum": float(np.min(activation_sum)),
        "median_activation_sum": float(np.median(activation_sum)),
        "min_abs_sigma": float(np.min(np.abs(model.sigmas))),
        "max_abs_sigma": float(np.max(np.abs(model.sigmas))),
        "max_abs_output": float(np.max(np.abs(pred))),
        "jacobian_norm_mean": float(np.mean(jac_norm)),
        "jacobian_norm_max": float(np.max(jac_norm)),
        "empirical_lipschitz_mean": float(np.mean(ratios)),
        "empirical_lipschitz_max": float(np.max(ratios)),
        "parameter_l2_norm": float(np.linalg.norm(vector)),
    }
