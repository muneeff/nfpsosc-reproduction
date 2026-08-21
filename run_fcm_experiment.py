"""
Independent Experiment: Fuzzy C-Means (FCM) + TSK Model for Short Series Forecasting
"""

from pathlib import Path
import numpy as np
import pandas as pd
import time

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "real_short_series"
SELECTED_CSV = OUT_DIR / "selected_real_series.csv"


def pure_numpy_fcm(X: np.ndarray, c: int, m: float = 2.0, max_iter: int = 100, error: float = 1e-4):
    """
    خوارزمية Fuzzy C-Means (FCM) مبنية بـ NumPy الصرف بدون تبعات خارجية
    X: شكل البيانات (n_samples, n_features)
    c: عدد القواعد أو العناقيد
    """
    n_samples, n_features = X.shape
    # تهيئة مصفوفة الانتماء U عشوائياً وتطبيعها
    np.random.seed(42)
    U = np.random.rand(n_samples, c)
    U /= np.sum(U, axis=1, keepdims=True)
    
    V = np.zeros((c, n_features))
    
    for iteration in range(max_iter):
        U_old = U.copy()
        
        # 1. حساب مراكز العناقيد V
        um = U ** m
        denom = np.sum(um, axis=0, keepdims=True).T
        denom[denom == 0] = 1e-10
        V = np.dot(um.T, X) / denom
        
        # 2. حساب المسافات
        dist = np.zeros((n_samples, c))
        for i in range(c):
            dist[:, i] = np.linalg.norm(X - V[i], axis=1)
        dist = np.maximum(dist, 1e-10)
        
        # 3. تحديث مصفوفة الانتماء U
        power = 2.0 / (m - 1)
        inv_dist = (1.0 / dist) ** power
        U = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)
        
        if np.linalg.norm(U - U_old) < error:
            break
            
    return V, U


class FCMTSKModel:
    """
    نموذج TSK ضبابي مهيأ بواسطة Fuzzy C-Means مع معلمات تابعة محسوبة عبر Ridge Regression
    """
    def __init__(self, n_lags: int, n_rules: int = 2, m: float = 2.0, alpha: float = 0.05):
        self.n_lags = n_lags
        self.n_rules = n_rules
        self.m = m
        self.alpha = alpha
        self.centers = None
        self.sigmas = None
        self.consequents = None
        self.scaler_mean = None
        self.scaler_scale = None

    def fit_scaler(self, data: np.ndarray):
        self.scaler_mean = np.mean(data)
        self.scaler_scale = np.std(data) + 1e-8

    def transform_scaler(self, data: np.ndarray):
        return (data - self.scaler_mean) / self.scaler_scale

    def inverse_scaler(self, data: np.ndarray):
        return data * self.scaler_scale + self.scaler_mean

    def _make_lags(self, series: np.ndarray):
        L = self.n_lags
        X = np.asarray([series[t - L : t] for t in range(L, len(series))], dtype=float)
        y = series[L:].copy()
        return X, y

    def fit(self, train_series: np.ndarray):
        self.fit_scaler(train_series)
        scaled = self.transform_scaler(train_series)
        X, y = self._make_lags(scaled)
        
        if len(X) <= self.n_rules:
            self.n_rules = max(1, len(X) - 1)

        # تطبيق FCM لاستخراج المراكز ومصفوفة الانتماء
        centers, U = pure_numpy_fcm(X, c=self.n_rules, m=self.m)
        self.centers = centers
        
        # حساب الانحرافات المعيارية (Sigmas) وزناً بالانتماء
        sigmas = np.zeros_like(centers)
        for i in range(self.n_rules):
            weights = U[:, i] ** self.m
            var = np.sum(weights[:, None] * (X - centers[i])**2, axis=0) / (np.sum(weights) + 1e-8)
            sigmas[i] = np.sqrt(var + 1e-5)
        self.sigmas = sigmas

        # حساب دالة التنشيط (Firing Strengths) وتصميم مصفوفة التصميم Phi
        Phi = self._compute_design_matrix(X)
        
        # حل المعلمات التابعة Consequents عبر Ridge Regression
        I_mat = np.eye(Phi.shape[1])
        self.consequents = np.linalg.inv(Phi.T @ Phi + self.alpha * I_mat) @ Phi.T @ y

    def _compute_design_matrix(self, X: np.ndarray):
        N, L = X.shape
        R = self.n_rules
        Phi_blocks = []
        
        for k in range(N):
            x = X[k]
            w = np.zeros(R)
            for i in range(R):
                # دالة العضوية الغوسية
                mu = np.exp(-0.5 * np.sum(((x - self.centers[i]) / (self.sigmas[i] + 1e-8))**2))
                w[i] = max(mu, 1e-8)
            
            w_sum = np.sum(w) + 1e-12
            w_norm = w / w_sum
            
            # بناء صف التصميم لكل قاعدة (1 + L معالم)
            row_elements = []
            for i in range(R):
                row_elements.append(w_norm[i])
                for j in range(L):
                    row_elements.append(w_norm[i] * x[j])
            Phi_blocks.append(row_elements)
            
        return np.asarray(Phi_blocks, dtype=float)

    def predict_single(self, lag_vector: np.ndarray):
        scaled_lag = self.transform_scaler(lag_vector).reshape(1, -1)
        Phi = self._compute_design_matrix(scaled_lag)
        pred_scaled = float(Phi @ self.consequents)
        return float(self.inverse_scaler(np.array([pred_scaled]))[0])

    def forecast(self, test_actuals: np.ndarray, full_history: np.ndarray):
        history = full_history.astype(float, copy=True)
        L = self.n_lags
        preds = np.empty(len(test_actuals), dtype=float)
        
        for idx, obs in enumerate(test_actuals):
            lag_vec = history[-L:]
            pred = self.predict_single(lag_vec)
            preds[idx] = pred
            history = np.append(history, float(obs))
        return preds


def compute_metrics(y_true, y_pred, train_data, m=1):
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    
    train_diff = train_data[m:] - train_data[:-m] if len(train_data) > m else train_data[1:] - train_data[:-1]
    scale = np.mean(np.abs(train_diff)) + 1e-12
    mase = float(mae / scale)
    return {"mase": mase, "rmse": rmse, "mae": mae}


def main():
    if not SELECTED_CSV.exists():
        print(f"Error: {SELECTED_CSV} not found. Run selection script first.")
        return

    selected = pd.read_csv(SELECTED_CSV)
    results = []

    print("="*60)
    print("Running Independent FCM-TSK Experiment on Real Short Series...")
    print("="*60)

    for _, row in selected.iterrows():
        dataset = str(row["series_id"])
        freq = str(row["frequency"]).lower()
        season_length = 12 if "month" in freq else (4 if "quarter" in freq else 1)
        
        # قراءة السلسلة
        path = ROOT / str(row["file_path"])
        df = pd.read_csv(path)
        col = "revenue_million_yer" if "revenue_million_yer" in df.columns else "value"
        values = df[col].dropna().to_numpy(dtype=float)

        # التقسيم (80% تدريب، 20% اختبار) مثل الباقين
        n_train = int(len(values) * 0.80)
        train_data = values[:n_train]
        test_data = values[n_train:]

        # بناء وتدريب نموذج FCM-TSK
        model = FCMTSKModel(n_lags=3, n_rules=2, alpha=0.05)
        model.fit(train_data)
        
        # التنبؤ
        preds = model.forecast(test_data, train_data)
        metrics = compute_metrics(test_data, preds, train_data, m=season_length)

        results.append({
            "dataset": dataset,
            "mase": metrics["mase"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"]
        })
        print(f"Dataset: {dataset:<20} | MASE: {metrics['mase']:.4f} | RMSE: {metrics['rmse']:.2f}")

    res_df = pd.DataFrame(results)
    print("="*60)
    print("FCM-TSK EXPERIMENT SUMMARY:")
    print(f"Mean MASE: {res_df['mase'].mean():.4f} (std: {res_df['mase'].std():.4f})")
    print(f"Mean RMSE: {res_df['rmse'].mean():.2f}")
    print(f"Mean MAE:  {res_df['mae'].mean():.2f}")
    print("="*60)


if __name__ == "__main__":
    main()