"""
Dynamic Radius Adaptive PSO for NFPSO-V2 (Mathematically Corrected Version)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class DynamicPSOParticle:
    """جسيم PSO مع دعم للترميز الديناميكي والمباشر"""
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
    محسن PSO مع تحسين ديناميكي لشعاع التجميع وتفعيل ضبابي حقيقي (Takagi-Sugeno)
    """
    
    def __init__(
        self,
        dim_features: int,
        R_max: int = 15,
        swarm_size: int = 12,
        max_iter: int = 40,
        radius_bounds: Tuple[float, float] = (0.15, 1.2),
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
        
        # أبعاد الجسيم: [r_c, C_1..C_Rmax, Sigma_1..Sigma_Rmax]
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
        """
        تطبيق Subtractive Clustering أصلي يعتمد على الجهد الرياضي (Numpy Pure)
        """
        N, D = X.shape
        if N == 0:
            return np.zeros((1, D)), np.ones((1, D))

        ra = float(r_c)
        rb = ra * 1.5
        
        # 1. حساب الجهد الابتدائي لكل نقطة P_i
        dist_matrix = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
        potentials = np.sum(np.exp(-dist_matrix / ((ra / 2.0) ** 2)), axis=1)
        
        centers = []
        sigmas = []
        
        p_max_first = np.max(potentials)
        if p_max_first <= 1e-12:
            return np.mean(X, axis=0, keepdims=True), np.ones((1, D)) * (ra / 2.0)

        # 2. استخراج المراكز طردياً
        for _ in range(self.R_max):
            max_idx = np.argmax(potentials)
            max_p = potentials[max_idx]
            
            if max_p < 0.15 * p_max_first:
                break
                
            center = X[max_idx].copy()
            centers.append(center)
            
            # تخفيض الجهد للنقاط المجاورة
            dists_to_center = np.sum((X - center) ** 2, axis=1)
            potentials -= max_p * np.exp(-dists_to_center / ((rb / 2.0) ** 2))
            potentials[potentials < 0] = 0

        if len(centers) == 0:
            centers = [np.mean(X, axis=0)]

        centers = np.array(centers)
        
        # 3. حساب عرض الدالة الجوسية (Sigma) لكل مركز
        for center in centers:
            dists = np.sqrt(np.sum((X - center) ** 2, axis=1))
            sig = np.median(dists) / np.sqrt(2.0)
            sig = max(sig, 1e-2)
            sigmas.append(np.full(D, sig))

        sigmas = np.array(sigmas)
        return centers, sigmas

    def _decode_particle(
        self, 
        position: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """فك تشفير الجسيم إلى r_c ومراكز وسيجما"""
        r_c = float(position[0])
        centers_raw = position[1:1 + self.R_max * self.dim].reshape(self.R_max, self.dim)
        sigmas_raw = position[1 + self.R_max * self.dim:1 + 2 * self.R_max * self.dim].reshape(
            self.R_max, self.dim
        )
        return r_c, centers_raw, sigmas_raw

    def _compute_firing_strengths(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray
    ) -> np.ndarray:
        """
        حساب درجات الانتماء التجميعية ومصفوفة التفعيل الفضفاضة (TSK Design Matrix)
        """
        N, D = X.shape
        R = centers.shape[0]
        
        # W_ir(x) = exp(-sum((x_d - m_rd)^2 / (2 * sigma_rd^2)))
        sigmas_clamped = np.maximum(sigmas, 1e-3)
        diff = (X[:, None, :] - centers[None, :, :]) ** 2  # (N, R, D)
        variances = 2.0 * (sigmas_clamped[None, :, :] ** 2)
        
        firing_weights = np.exp(-np.sum(diff / variances, axis=-1))  # (N, R)
        weight_sums = np.sum(firing_weights, axis=1, keepdims=True)  # (N, 1)
        weight_sums[weight_sums < 1e-12] = 1e-12
        
        normalized_w = firing_weights / weight_sums  # (N, R)
        
        # إنشاء مصفوفة TSK Consequent: [w_1*1, w_1*x_1..w_1*x_D, ..., w_R*1, w_R*x_1..w_R*x_D]
        X_design = np.hstack([np.ones((N, 1)), X])  # (N, D+1)
        Phi_list = []
        for r in range(R):
            w_r = normalized_w[:, r:r+1]  # (N, 1)
            Phi_r = w_r * X_design  # (N, D+1)
            Phi_list.append(Phi_r)
            
        Phi = np.hstack(Phi_list)  # (N, R*(D+1))
        return Phi

    def _compute_fitness(
        self,
        position: np.ndarray,
        mask: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """حساب دالة الهدف بناءً على نموذج Takagi-Sugeno الموزون فعلياً"""
        r_c, centers, sigmas = self._decode_particle(position)
        
        active_indices = np.where(mask)[0]
        R_active = len(active_indices)
        
        if R_active == 0:
            return 1e6
            
        centers_active = centers[active_indices]
        sigmas_active = sigmas[active_indices]
        
        try:
            from sklearn.linear_model import Ridge
            
            # بناء الفضاء الضبابي
            Phi_train = self._compute_firing_strengths(X_train, centers_active, sigmas_active)
            Phi_val = self._compute_firing_strengths(X_val, centers_active, sigmas_active)
            
            model = Ridge(alpha=self.alpha, fit_intercept=False)
            model.fit(Phi_train, y_train)
            
            y_pred = model.predict(Phi_val)
            rmse = float(np.sqrt(np.mean((y_val - y_pred) ** 2)))
            
            # عقوبة المعقدية لضبط عدد القواعد
            penalty = self.penalty_lambda * (R_active / float(self.R_max))
            
            if not np.isfinite(rmse):
                return 1e6
                
            fitness = rmse + penalty
            
        except Exception as e:
            logger.debug(f"Fitness evaluation failed: {e}")
            fitness = 1e6
            
        return float(fitness)

    def initialize_swarm(self, X_train: np.ndarray) -> None:
        """تهيئة السرب بالقيم الابتدائية وتعيين Mask الصحيح"""
        r_c_initial = 0.5
        centers_initial, sigmas_initial = self._subtractive_clustering(X_train, r_c_initial)
        R_initial = min(len(centers_initial), self.R_max)
        
        self.swarm = []
        
        for i in range(self.swarm_size):
            position = np.zeros(self.particle_dim)
            position[0] = np.clip(r_c_initial + 0.05 * np.random.randn(), *self.radius_bounds)
            mask = np.zeros(self.R_max, dtype=bool)
            mask[:R_initial] = True
            
            for r in range(self.R_max):
                if r < R_initial:
                    pos_start = 1 + r * self.dim
                    pos_end = 1 + (r + 1) * self.dim
                    position[pos_start:pos_end] = centers_initial[r] + 0.02 * np.random.randn(self.dim)
                    
                    sig_start = 1 + self.R_max * self.dim + r * self.dim
                    sig_end = 1 + self.R_max * self.dim + (r + 1) * self.dim
                    position[sig_start:sig_end] = sigmas_initial[r] + 0.01 * np.abs(np.random.randn(self.dim))
            
            particle = DynamicPSOParticle(
                position=position.copy(),
                velocity=np.zeros(self.particle_dim),
                best_position=position.copy(),
                best_fitness=np.inf,
                mask=mask,
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
        """تشغيل خوارزمية PSO المحسنة والمحققة رياضياتياً"""
        self.initialize_swarm(X_train)
        
        for iteration in range(self.max_iter):
            # إعادة التجميع الدوري بتغير r_c لكل جسيم
            if iteration % self.recompute_interval == 0 and iteration > 0:
                for particle in self.swarm:
                    r_c = particle.position[0]
                    try:
                        centers_new, sigmas_new = self._subtractive_clustering(X_train, r_c)
                        R_new = min(len(centers_new), self.R_max)
                        
                        particle.mask = np.zeros(self.R_max, dtype=bool)
                        particle.mask[:R_new] = True
                        particle.R_current = R_new
                        
                        for r in range(self.R_max):
                            pos_start = 1 + r * self.dim
                            pos_end = 1 + (r + 1) * self.dim
                            sig_start = 1 + self.R_max * self.dim + r * self.dim
                            sig_end = 1 + self.R_max * self.dim + (r + 1) * self.dim
                            
                            if r < R_new:
                                particle.position[pos_start:pos_end] = centers_new[r]
                                particle.position[sig_start:sig_end] = sigmas_new[r]
                            else:
                                particle.position[pos_start:pos_end] = 0.0
                                particle.position[sig_start:sig_end] = 0.0
                                particle.velocity[pos_start:pos_end] = 0.0
                                particle.velocity[sig_start:sig_end] = 0.0
                    except Exception as e:
                        logger.warning(f"Re-clustering failed at iter {iteration}: {e}")

            # تقييم اللياقة وتحديث Best
            for particle in self.swarm:
                fitness = self._compute_fitness(
                    particle.position, particle.mask,
                    X_train, y_train, X_val, y_val
                )
                
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()
                    
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
                    self.global_best_r_c = particle.position[0]
                    self.global_best_R = particle.R_current

            # تحديث السرعة والموقع بـ Projected Constriction Mechanism
            for particle in self.swarm:
                r1 = np.random.rand(self.particle_dim)
                r2 = np.random.rand(self.particle_dim)
                
                velocity = self.chi * (
                    self.omega * particle.velocity +
                    self.c1 * r1 * (particle.best_position - particle.position) +
                    self.c2 * r2 * (self.global_best_position - particle.position)
                )
                
                new_position = particle.position + velocity
                
                # Projection Operator on Bounds
                new_position[0] = np.clip(new_position[0], *self.radius_bounds)
                
                # تصفير الأبعاد الخاملة
                for r in range(self.R_max):
                    if not particle.mask[r]:
                        pos_start = 1 + r * self.dim
                        pos_end = 1 + (r + 1) * self.dim
                        sig_start = 1 + self.R_max * self.dim + r * self.dim
                        sig_end = 1 + self.R_max * self.dim + (r + 1) * self.dim
                        
                        new_position[pos_start:pos_end] = 0.0
                        new_position[sig_start:sig_end] = 0.0
                        velocity[pos_start:pos_end] = 0.0
                        velocity[sig_start:sig_end] = 0.0
                
                particle.velocity = velocity
                particle.position = new_position

            self.fitness_history.append(self.global_best_fitness)
            self.R_history.append(self.global_best_R)

        if self.global_best_position is not None:
            best_r_c, best_centers_raw, best_sigmas_raw = self._decode_particle(self.global_best_position)
            
            # إعادة التجميع النهائي لأفضل r_c
            best_centers, best_sigmas = self._subtractive_clustering(X_train, best_r_c)
            best_R = len(best_centers)
            
            return {
                'best_position': self.global_best_position.copy(),
                'best_fitness': self.global_best_fitness,
                'best_r_c': best_r_c,
                'best_R': best_R,
                'best_centers': best_centers,
                'best_sigmas': best_sigmas,
                'fitness_history': self.fitness_history,
                'R_history': self.R_history,
            }
        else:
            return {
                'best_position': None,
                'best_fitness': np.inf,
                'best_r_c': 0.5,
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