"""
Dynamic Radius Adaptive PSO for NFPSO-V2
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class DynamicPSOParticle:
    """جسيم PSO مع دعم للترميز الديناميكي"""
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float = np.inf
    mask: np.ndarray = field(default_factory=lambda: np.ones(15, dtype=bool))
    r_c: float = 1.0
    R_current: int = 0
    centers: np.ndarray = field(default_factory=lambda: np.array([]))
    sigmas: np.ndarray = field(default_factory=lambda: np.array([]))


class DynamicRadiusPSO:
    """
    محسن PSO مع تحسين ديناميكي لشعاع التجميع
    """
    
    def __init__(
        self,
        dim_features: int,
        R_max: int = 15,
        swarm_size: int = 12,
        max_iter: int = 40,
        radius_bounds: Tuple[float, float] = (0.1, 1.5),
        recompute_interval: int = 5,
        alpha: float = 0.05,
        penalty_lambda: float = 0.01,
        omega: float = 0.729,
        c1: float = 1.494,
        c2: float = 1.494,
        chi: float = 0.729,
    ):
        self.dim = dim_features
        self.R_max = R_max
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.radius_bounds = radius_bounds
        self.recompute_interval = recompute_interval
        self.alpha = alpha
        self.penalty_lambda = penalty_lambda
        
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.chi = chi
        
        self.particle_dim = 1 + 2 * self.R_max * self.dim
        
        self.swarm: List[DynamicPSOParticle] = []
        self.global_best_position: Optional[np.ndarray] = None
        self.global_best_fitness = np.inf
        self.global_best_centers: Optional[np.ndarray] = None
        self.global_best_sigmas: Optional[np.ndarray] = None
        self.global_best_r_c: float = 1.0
        self.global_best_R: int = 0
        
        self.fitness_history: List[float] = []
        self.R_history: List[int] = []
        
    def _subtractive_clustering(
        self, 
        X: np.ndarray, 
        r_c: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """تطبيق Subtractive Clustering"""
        from sklearn.cluster import MeanShift
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        bandwidth = r_c * 0.5
        ms = MeanShift(bandwidth=bandwidth, min_bin_freq=1)
        ms.fit(X_scaled)
        
        centers = scaler.inverse_transform(ms.cluster_centers_)
        
        sigmas = []
        for center in centers:
            distances = np.linalg.norm(X - center, axis=1)
            sigma = np.median(distances) * 0.5
            sigmas.append(max(sigma, 0.01))
        
        sigmas = np.array(sigmas).reshape(-1, 1)
        
        if len(centers) == 0:
            centers = np.mean(X, axis=0).reshape(1, -1)
            sigmas = np.array([0.5]).reshape(1, 1)
        
        if len(centers) > self.R_max:
            indices = np.argsort(
                [np.sum(np.linalg.norm(X - c, axis=1)) for c in centers]
            )[:self.R_max]
            centers = centers[indices]
            sigmas = sigmas[indices]
        
        return centers, sigmas
    
    def _decode_particle(
        self, 
        position: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """فك تشفير الجسيم"""
        r_c = position[0]
        centers = position[1:1 + self.R_max * self.dim].reshape(self.R_max, self.dim)
        sigmas = position[1 + self.R_max * self.dim:1 + 2 * self.R_max * self.dim].reshape(
            self.R_max, self.dim
        )
        return r_c, centers, sigmas
    
    def _get_active_mask(self, centers: np.ndarray) -> np.ndarray:
        """تحديد القواعد النشطة"""
        norms = np.linalg.norm(centers, axis=1)
        return norms > 1e-6
    
    def _get_active_count(self, centers: np.ndarray) -> int:
        return int(np.sum(self._get_active_mask(centers)))
    
    def _build_fuzzy_model(
        self,
        centers: np.ndarray,
        sigmas: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        alpha: float
    ):
        """بناء نموذج Takagi-Sugeno"""
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        return model
    
    def _compute_fitness(
        self,
        position: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """حساب دالة الهدف"""
        r_c, centers, sigmas = self._decode_particle(position)
        active = self._get_active_mask(centers)
        R_active = int(np.sum(active))
        
        if R_active == 0:
            return 1e6
        
        centers_active = centers[active]
        sigmas_active = sigmas[active]
        
        try:
            model = self._build_fuzzy_model(
                centers_active, sigmas_active,
                X_train, y_train,
                self.alpha
            )
            
            y_pred = model.predict(X_val)
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            penalty = self.penalty_lambda * (R_active / self.R_max)
            
            if R_active > 1:
                center_distances = np.linalg.norm(
                    centers_active[:, None, :] - centers_active[None, :, :], 
                    axis=2
                )
                np.fill_diagonal(center_distances, np.inf)
                min_distance = np.min(center_distances)
                if min_distance < 0.01:
                    penalty += 10.0
            
            fitness = rmse + penalty
            
        except Exception as e:
            logger.warning(f"Fitness computation failed: {e}")
            fitness = 1e6
        
        return float(fitness)
    
    def initialize_swarm(self, X_train: np.ndarray) -> None:
        """تهيئة السرب"""
        r_c_initial = 1.0
        centers_initial, sigmas_initial = self._subtractive_clustering(
            X_train, r_c_initial
        )
        R_initial = len(centers_initial)
        
        self.swarm = []
        
        for i in range(self.swarm_size):
            position = np.zeros(self.particle_dim)
            position[0] = np.clip(r_c_initial + 0.1 * np.random.randn(), *self.radius_bounds)
            
            for r in range(self.R_max):
                if r < R_initial:
                    pos_start = 1 + r * self.dim
                    pos_end = 1 + (r + 1) * self.dim
                    position[pos_start:pos_end] = (
                        centers_initial[r % len(centers_initial)] + 
                        0.05 * np.random.randn(self.dim)
                    )
                    
                    sig_start = 1 + self.R_max * self.dim + r * self.dim
                    sig_end = 1 + self.R_max * self.dim + (r + 1) * self.dim
                    position[sig_start:sig_end] = (
                        sigmas_initial[r % len(sigmas_initial)].flatten() + 
                        0.05 * np.abs(np.random.randn(self.dim))
                    )
                else:
                    pos_start = 1 + r * self.dim
                    pos_end = 1 + (r + 1) * self.dim
                    position[pos_start:pos_end] = 0.1 * np.random.randn(self.dim)
                    
                    sig_start = 1 + self.R_max * self.dim + r * self.dim
                    sig_end = 1 + self.R_max * self.dim + (r + 1) * self.dim
                    position[sig_start:sig_end] = np.abs(0.1 * np.random.randn(self.dim)) + 0.01
            
            particle = DynamicPSOParticle(
                position=position.copy(),
                velocity=np.zeros(self.particle_dim),
                best_position=position.copy(),
                best_fitness=np.inf,
                mask=self._get_active_mask(
                    position[1:1 + self.R_max * self.dim].reshape(self.R_max, self.dim)
                ),
                r_c=position[0],
                R_current=R_initial,
            )
            self.swarm.append(particle)
        
        self.global_best_position = self.swarm[0].position.copy()
        self.global_best_fitness = np.inf
        
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """تشغيل خوارزمية PSO المحسّنة"""
        self.initialize_swarm(X_train)
        
        for iteration in range(self.max_iter):
            if iteration % self.recompute_interval == 0 and iteration > 0:
                for particle in self.swarm:
                    r_c = particle.position[0]
                    try:
                        centers_new, sigmas_new = self._subtractive_clustering(
                            X_train, r_c
                        )
                        R_new = len(centers_new)
                        
                        particle.mask[:R_new] = True
                        particle.mask[R_new:] = False
                        particle.R_current = R_new
                        
                        for r in range(min(R_new, self.R_max)):
                            pos_start = 1 + r * self.dim
                            pos_end = 1 + (r + 1) * self.dim
                            particle.position[pos_start:pos_end] = centers_new[r]
                            
                            sig_start = 1 + self.R_max * self.dim + r * self.dim
                            sig_end = 1 + self.R_max * self.dim + (r + 1) * self.dim
                            particle.position[sig_start:sig_end] = sigmas_new[r].flatten()
                    except Exception as e:
                        logger.warning(f"Re-clustering failed at iteration {iteration}: {e}")
            
            for particle in self.swarm:
                fitness = self._compute_fitness(
                    particle.position,
                    X_train, y_train,
                    X_val, y_val
                )
                
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()
                
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
            
            for particle in self.swarm:
                r1 = np.random.rand(self.particle_dim)
                r2 = np.random.rand(self.particle_dim)
                
                velocity = (
                    self.chi * (
                        self.omega * particle.velocity +
                        self.c1 * r1 * (particle.best_position - particle.position) +
                        self.c2 * r2 * (self.global_best_position - particle.position)
                    )
                )
                
                new_position = particle.position + velocity
                new_position[0] = np.clip(new_position[0], *self.radius_bounds)
                
                centers = new_position[1:1 + self.R_max * self.dim].reshape(self.R_max, self.dim)
                centers = np.clip(centers, -10, 10)
                new_position[1:1 + self.R_max * self.dim] = centers.flatten()
                
                sigmas = new_position[1 + self.R_max * self.dim:1 + 2 * self.R_max * self.dim].reshape(
                    self.R_max, self.dim
                )
                sigmas = np.abs(sigmas) + 0.01
                new_position[1 + self.R_max * self.dim:1 + 2 * self.R_max * self.dim] = sigmas.flatten()
                
                particle.velocity = velocity
                particle.position = new_position
            
            self.fitness_history.append(self.global_best_fitness)
            if self.global_best_position is not None:
                _, centers, _ = self._decode_particle(self.global_best_position)
                self.R_history.append(self._get_active_count(centers))
            
            logger.info(f"Iteration {iteration + 1}/{self.max_iter}, Best fitness: {self.global_best_fitness:.4f}")
        
        if self.global_best_position is not None:
            best_r_c, best_centers, best_sigmas = self._decode_particle(
                self.global_best_position
            )
            active = self._get_active_mask(best_centers)
            best_R = int(np.sum(active))
            
            return {
                'best_position': self.global_best_position.copy(),
                'best_fitness': self.global_best_fitness,
                'best_r_c': best_r_c,
                'best_R': best_R,
                'best_centers': best_centers[active].copy(),
                'best_sigmas': best_sigmas[active].copy(),
                'fitness_history': self.fitness_history,
                'R_history': self.R_history,
            }
        else:
            return {
                'best_position': None,
                'best_fitness': np.inf,
                'best_r_c': 1.0,
                'best_R': 0,
                'best_centers': np.array([]),
                'best_sigmas': np.array([]),
                'fitness_history': [],
                'R_history': [],
            }


__all__ = [
    'DynamicRadiusPSO',
    'DynamicPSOParticle',
]