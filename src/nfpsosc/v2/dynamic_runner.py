"""
Dynamic Runner for NFPSO-V2 - متوافق مع بنية TSKModel الصحيحة
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

from .dynamic_radius_pso import DynamicRadiusPSO
from .pc_nfpso import prepare_pc_nfpso_training_data
from .model import TSKModel
from .initialization import initialize_tsk_from_subtractive_clustering

logger = logging.getLogger(__name__)


class DynamicPCNFPSO:
    """
    نسخة مطورة من PC-NFPSO مع تحسين ديناميكي لـ r_c
    """
    
    def __init__(
        self,
        n_lags: int,
        validation_size: int,
        R_max: int = 15,
        swarm_size: int = 12,
        max_iter: int = 40,
        radius_bounds: Tuple[float, float] = (0.1, 1.5),
        recompute_interval: int = 5,
        alpha: float = 0.05,
        penalty_lambda: float = 0.01,
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
        self.initialization = None
        
    def fit(
        self, 
        raw_pretest: np.ndarray,
    ) -> Dict[str, Any]:
        """
        تدريب النموذج باستخدام DynamicRadiusPSO
        """
        logger.info("Training DynamicPCNFPSO...")
        
        # 1. تحضير بيانات التدريب
        self.training_data = prepare_pc_nfpso_training_data(
            raw_pretest,
            n_lags=self.n_lags,
            validation_size=self.validation_size,
        )
        
        # 2. تهيئة النموذج باستخدام Subtractive Clustering مع r_c = 1.0
        self.initialization = initialize_tsk_from_subtractive_clustering(
            self.training_data.X_fit,
            self.training_data.y_fit,
            radius=1.0,
        )
        
        # 3. تهيئة المحسن الديناميكي
        self.optimizer = DynamicRadiusPSO(
            dim_features=self.n_lags,
            R_max=self.R_max,
            swarm_size=self.swarm_size,
            max_iter=self.max_iter,
            radius_bounds=self.radius_bounds,
            recompute_interval=self.recompute_interval,
            alpha=self.alpha,
            penalty_lambda=self.penalty_lambda
        )
        
        # 4. تشغيل التحسين
        self.best_result = self.optimizer.optimize(
            self.training_data.X_fit,
            self.training_data.y_fit,
            self.training_data.X_validation,
            self.training_data.y_validation
        )
        
        # 5. بناء النموذج النهائي باستخدام TSKModel
        if self.best_result is not None:
            R = self.best_result['best_R']
            centers = self.best_result['best_centers']
            sigmas = self.best_result['best_sigmas']
            
            # إنشاء TSKModel بالشكل الصحيح
            # نحتاج إلى تهيئة المتغيرات التابعة (consequents) بقيم صفرية
            # وسيتم تدريبها لاحقاً باستخدام fit_consequents
            n_features = self.n_lags
            # المتغيرات التابعة: لكل قاعدة (n_features + 1) معامل (بما في ذلك الثابت)
            consequents = np.zeros((R, n_features + 1))
            
            self.final_model = TSKModel(
                centers=centers,
                sigmas=sigmas,
                consequents=consequents
            )
            
            # تدريب النموذج على كامل بيانات ما قبل الاختبار
            self.final_model.fit_consequents(
                self.training_data.X_pretest,
                self.training_data.y_pretest,
                alpha=self.alpha,
            )
        
        logger.info(f"Training complete: r_c={self.best_result['best_r_c']:.4f}, R={self.best_result['best_R']}")
        
        return self.best_result
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """التنبؤ باستخدام النموذج المُدرّب"""
        if self.final_model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        return self.final_model.predict(X)
    
    def forecast(self, test_actuals: np.ndarray) -> np.ndarray:
        """التنبؤ خطوة بخطوة مع تحديث التاريخ"""
        if self.final_model is None or self.training_data is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        actuals = np.asarray(test_actuals, dtype=float).reshape(-1)
        history = self.training_data.raw_pretest.astype(float, copy=True)
        L = self.n_lags
        
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
        """الحصول على تكوين النموذج"""
        return {
            'method': 'pc_nfpso_dynamic',
            'n_lags': self.n_lags,
            'validation_size': self.validation_size,
            'R_max': self.R_max,
            'swarm_size': self.swarm_size,
            'max_iter': self.max_iter,
            'radius_bounds': self.radius_bounds,
            'recompute_interval': self.recompute_interval,
            'alpha': self.alpha,
            'penalty_lambda': self.penalty_lambda,
            'optimizer_seed': self.optimizer_seed,
            'best_r_c': self.best_result['best_r_c'] if self.best_result else None,
            'best_R': self.best_result['best_R'] if self.best_result else None,
            'best_fitness': self.best_result['best_fitness'] if self.best_result else None,
        }


__all__ = ['DynamicPCNFPSO']