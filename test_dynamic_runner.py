"""
اختبار DynamicPCNFPSO المتوافق مع pc_nfpso.py
"""

import numpy as np
from nfpsosc.v2.dynamic_runner import DynamicPCNFPSO

print('📊 Testing DynamicPCNFPSO...')

# بيانات تجريبية
np.random.seed(42)
n_samples = 200
raw_series = np.cumsum(np.random.randn(n_samples)) + 10

# تهيئة النموذج
model = DynamicPCNFPSO(
    n_lags=5,
    validation_size=20,
    swarm_size=6,
    max_iter=5,
    recompute_interval=2
)

# تدريب
result = model.fit(raw_series)

print(f'✅ Best r_c: {result["best_r_c"]:.4f}')
print(f'✅ Best R: {result["best_R"]}')
print(f'✅ Best fitness: {result["best_fitness"]:.4f}')
print(f'✅ R history: {result["R_history"]}')
print('🎉 Test completed!')