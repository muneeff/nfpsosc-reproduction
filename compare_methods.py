"""
مقارنة بين PC-NFPSO التقليدي و DynamicPCNFPSO
"""

import numpy as np
from nfpsosc.v2.dynamic_runner import DynamicPCNFPSO
from nfpsosc.v2.pc_nfpso import fit_pc_nfpso_v2

print('📊 مقارنة بين PC-NFPSO التقليدي و DynamicPCNFPSO...')
np.random.seed(42)

# بيانات تجريبية
raw_series = np.cumsum(np.random.randn(200)) + 10

# PC-NFPSO التقليدي
print('🔄 تشغيل PC-NFPSO التقليدي...')
result_std = fit_pc_nfpso_v2(
    raw_pretest=raw_series,
    n_lags=5,
    validation_size=20,
    radius=1.0,
    alpha=0.05,
    optimizer_seed=42
)
print(f'   R: {len(result_std.final_model.centers)}')
print(f'   Cost: {result_std.optimizer_result.best_cost:.4f}')

# Dynamic PC-NFPSO
print('🔄 تشغيل DynamicPCNFPSO...')
model_dyn = DynamicPCNFPSO(
    n_lags=5,
    validation_size=20,
    swarm_size=6,
    max_iter=5
)
result_dyn = model_dyn.fit(raw_series)
print(f'   r_c: {result_dyn["best_r_c"]:.4f}')
print(f'   R: {result_dyn["best_R"]}')
print(f'   Fitness: {result_dyn["best_fitness"]:.4f}')

print('✅ المقارنة مكتملة!')