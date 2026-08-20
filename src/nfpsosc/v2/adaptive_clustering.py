"""
Adaptive Clustering for Dynamic NFPSO
"""

from __future__ import annotations

import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


def subtractive_clustering(
    X: np.ndarray,
    r_c: float = 1.0,
    min_density: float = 0.5,
    max_clusters: int = 15,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """خوارزمية Subtractive Clustering"""
    if len(X) == 0:
        raise ValueError("Empty input data")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_samples = X_scaled.shape[0]
    r_c_scaled = r_c * 0.5
    
    densities = np.zeros(n_samples)
    for i in range(n_samples):
        diff = X_scaled - X_scaled[i]
        distances = np.sqrt(np.sum(diff ** 2, axis=1))
        densities[i] = np.sum(np.exp(- (distances ** 2) / (2 * (r_c_scaled ** 2))))
    
    centers = []
    sigmas = []
    
    for _ in range(max_clusters):
        max_density_idx = np.argmax(densities)
        max_density = densities[max_density_idx]
        
        if max_density < min_density * np.max(densities):
            break
        
        center = X[max_density_idx]
        centers.append(center)
        
        diff = X - center
        distances = np.sqrt(np.sum(diff ** 2, axis=1))
        sigma = np.median(distances) * 0.5
        sigmas.append(max(sigma, 0.01))
        
        r_c_reduction = r_c_scaled * 1.5
        for i in range(n_samples):
            diff_i = X_scaled[i] - X_scaled[max_density_idx]
            dist_i = np.sqrt(np.sum(diff_i ** 2))
            reduction = np.exp(- (dist_i ** 2) / (2 * (r_c_reduction ** 2)))
            densities[i] -= max_density * reduction
    
    centers = np.array(centers)
    sigmas = np.array(sigmas).reshape(-1, 1)
    
    if len(centers) == 0:
        centers = np.mean(X, axis=0).reshape(1, -1)
        sigmas = np.array([np.std(X, axis=0).mean()]).reshape(1, 1)
    
    if len(centers) > max_clusters:
        weights = [np.sum(np.exp(-np.linalg.norm(X - c, axis=1))) for c in centers]
        indices = np.argsort(weights)[-max_clusters:]
        centers = centers[indices]
        sigmas = sigmas[indices]
    
    if verbose:
        logger.info(f"Subtractive Clustering: found {len(centers)} clusters with r_c={r_c}")
    
    return centers, sigmas


def adaptive_recluster(
    X: np.ndarray,
    current_centers: np.ndarray,
    current_sigmas: np.ndarray,
    r_c: float,
    R_max: int = 15
) -> Tuple[np.ndarray, np.ndarray, int]:
    """إعادة التجميع التكيفي"""
    new_centers, new_sigmas = subtractive_clustering(
        X, r_c=r_c, max_clusters=R_max
    )
    R = len(new_centers)
    
    if R == 0:
        return current_centers, current_sigmas, len(current_centers)
    
    return new_centers, new_sigmas, R


def compute_cluster_quality(
    X: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray
) -> float:
    """حساب جودة التجميع"""
    if len(centers) == 0:
        return 0.0
    
    within_cluster_sum = 0.0
    for i, center in enumerate(centers):
        diff = X - center
        distances = np.sqrt(np.sum(diff ** 2, axis=1))
        within_cluster_sum += np.sum(distances * np.exp(-distances / (2 * sigmas[i])))
    
    between_cluster_sum = 0.0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = np.linalg.norm(centers[i] - centers[j])
            between_cluster_sum += dist
    
    if between_cluster_sum < 1e-6:
        return 0.0
    
    quality = within_cluster_sum / (between_cluster_sum + 1e-6)
    return -quality


__all__ = [
    'subtractive_clustering',
    'adaptive_recluster',
    'compute_cluster_quality',
]