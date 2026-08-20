"""
اختبار التنبؤ باستخدام DynamicPCNFPSO
"""

import numpy as np
from nfpsosc.v2.dynamic_runner import DynamicPCNFPSO

print('📊 اختبار التنبؤ باستخدام DynamicPCNFPSO...')

# بيانات تجريبية
np.random.seed(42)
n_train = 150
n_test = 30
raw_series = np.cumsum(np.random.randn(n_train + n_test)) + 10

# تقسيم البيانات
train_data = raw_series[:n_train]
test_data = raw_series[n_train:]

# تدريب النموذج
model = DynamicPCNFPSO(
    n_lags=5,
    validation_size=20,
    swarm_size=6,
    max_iter=5
)
result = model.fit(train_data)
print(f'✅ Training complete: r_c={result["best_r_c"]:.4f}, R={result["best_R"]}')

# التنبؤ
predictions = model.forecast(test_data)
print(f'✅ Predictions shape: {predictions.shape}')

# حساب الخطأ
rmse = np.sqrt(np.mean((predictions - test_data) ** 2))
mae = np.mean(np.abs(predictions - test_data))

print(f'   RMSE: {rmse:.4f}')
print(f'   MAE: {mae:.4f}')
print('🎉 اختبار التنبؤ مكتمل!')