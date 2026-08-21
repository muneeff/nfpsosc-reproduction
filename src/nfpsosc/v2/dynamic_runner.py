"""
Dynamic Runner for NFPSO-V2 - النسخة المتقدمة المحسنة للحد الأدنى من الأخطاء (MASE)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

from .dynamic_radius_pso import DynamicRadiusPSO
from .pc_nfpso import prepare_pc_nfpso_training_data
from .model import TSKModel

logger = logging.getLogger(__name__)


def optimized_numpy_fcm(X: np.ndarray, c: int, m: float = 1.3, max_iter: int = 150, error: float = 1e-5):
    """خوارزمية Fuzzy C-Means مع Fuzzifier حاد (m=1.3) لزيادة دقة التقسيم المحلي"""
    n_samples, n_features = X.shape
    np.random.seed(42)
    
    U = np.random.rand(n_samples, c)
    U /= np.sum(U, axis=1, keepdims=True)
    V = np.zeros((c, n_features))
    
    for iteration in range(max_iter):
        U_old = U.copy()
        um = U ** m
        denom = np.sum(um, axis=0, keepdims=True).T
        denom[denom == 0] = 1e-10
        V = np.dot(um.T, X) / denom
        
        dist = np.zeros((n_samples, c))
        for i in range(c):
            dist[:, i] = np.linalg.norm(X - V[i], axis=1)
        dist = np.maximum(dist, 1e-10)
        
        power = 2.0 / (m - 1)
        inv_dist = (1.0 / dist) ** power
        U = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)
        
        if np.linalg.norm(U - U_old) < error:
            break
            
    return V, U


class DynamicPCNFPSO:
    """
    نسخة مطورة متطرفة الاستقرار تعتمد على FCM الحادة وعقاب صارم للتعقيد
    """
    
    def __init__(
        self,
        n_lags: int,
        validation_size: int,
        R_max: int = 2, # تقييد أقصى عدد للقواعد لتعزيز الاستقرار في السلاسل الشحيحة
        swarm_size: int = 15,
        max_iter: int = 45,
        radius_bounds: Tuple[float, float] = (0.20, 0.60),
        recompute_interval: int = 5,
        alpha: float = 0.05,
        penalty_lambda: float = 0.02, # عقوبة أعلى لمنع القواعد الزائدة
        optimizer_seed: int = 42,
        **kwargs
    ):
        self.n_lags = n_lags
        self.validation_size = validation_size
        self.R_max = R_max
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.radius_bounds = radius_bounds
        self.recompute_interval = recompute_interval
        self.alpha = alpha
        self.penalty_lambda = penalty_lambda
        self.optimizer_seed = optimizer_seed
        self.kwargs = kwargs
        
        self.optimizer = None
        self.best_result = None
        self.training_data = None
        self.final_model = None
        self.effective_lags = n_lags
        
    def fit(self, raw_pretest: np.ndarray) -> Dict[str, Any]:
        logger.info("Training DynamicPCNFPSO with Ultra-Optimized FCM...")
        
        series = np.asarray(raw_pretest, dtype=float).reshape(-1)
        
        if len(series) < 25:
            self.effective_lags = min(self.n_lags, 2)
        else:
            self.effective_lags = self.n_lags
            
        self.training_data = prepare_pc_nfpso_training_data(
            series,
            n_lags=self.effective_lags,
            validation_size=self.validation_size,
        )
        
        X_fit = self.training_data.X_fit
        y_fit = self.training_data.y_fit
        
        n_rules = min(self.R_max, max(1, len(X_fit) // 6))
        
        centers, U = optimized_numpy_fcm(X_fit, c=n_rules, m=1.3)
        
        sigmas = np.zeros_like(centers)
        for i in range(n_rules):
            weights = U[:, i] ** 1.3
            var = np.sum(weights[:, None] * (X_fit - centers[i])**2, axis=0) / (np.sum(weights) + 1e-8)
            sigmas[i] = np.maximum(np.sqrt(var + 1e-4), 0.12)

        self.optimizer = DynamicRadiusPSO(
            dim_features=self.effective_lags,
            R_max=n_rules,
            swarm_size=self.swarm_size,
            max_iter=self.max_iter,
            radius_bounds=self.radius_bounds,
            recompute_interval=self.recompute_interval,
            alpha=self.alpha,
            penalty_lambda=self.penalty_lambda
        )
        
        self.best_result = self.optimizer.optimize(
            X_fit,
            y_fit,
            self.training_data.X_validation,
            self.training_data.y_validation
        )
        
        if self.best_result is not None:
            R = self.best_result['best_R']
            opt_centers = self.best_result['best_centers'][:R]
            opt_sigmas = self.best_result['best_sigmas'][:R]
            
            consequents = np.zeros((R, self.effective_lags + 1))
            self.final_model = TSKModel(
                centers=opt_centers,
                sigmas=opt_sigmas,
                consequents=consequents
            )
            
            self.final_model.fit_consequents(
                self.training_data.X_pretest,
                self.training_data.y_pretest,
                alpha=self.alpha,
            )
        
        return self.best_result
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.final_model is None:
            raise ValueError("Model not trained yet.")
        return self.final_model.predict(X)
    
    def forecast(self, test_actuals: np.ndarray) -> np.ndarray:
        if self.final_model is None or self.training_data is None:
            raise ValueError("Model not trained yet.")
        
        actuals = np.asarray(test_actuals, dtype=float).reshape(-1)
        history = self.training_data.raw_pretest.astype(float, copy=True)
        L = self.effective_lags
        
        predictions = np.empty(len(actuals), dtype=float)
        for i, observed in enumerate(actuals):
            lag_raw = history[-L:]
            lag_scaled = self.training_data.scaler.transform(lag_raw).reshape(1, -1)
            pred_scaled = float(self.final_model.predict(lag_scaled)[0])
            pred_raw = float(
                self.training_data.scaler.inverse_transform(
                    np.asarray([pred_scaled])
                )[0]
            )
            predictions[i] = pred_raw
            history = np.append(history, float(observed))
        
        return predictions
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'method': 'ultra_fcm_nfpso_dynamic',
            'n_lags': self.effective_lags,
            'validation_size': self.validation_size,
            'best_R': self.best_result['best_R'] if self.best_result else None,
        }


__all__ = ['DynamicPCNFPSO']