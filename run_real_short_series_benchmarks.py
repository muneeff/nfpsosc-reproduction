"""
Optimized Hybrid Residual Benchmarking Script: Linear Baseline + Pure Residual NFPSO
"""

from pathlib import Path
import pandas as pd
import numpy as np
import time
import argparse
from typing import Tuple

from src.nfpsosc.v2.dynamic_runner import DynamicPCNFPSO

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "real_short_series"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SELECTED_CSV = OUT_DIR / "selected_real_series.csv"


def get_linear_trend(train_data: np.ndarray, total_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """حساب الاتجاه الخطي للتدريب والاختبار بدقة"""
    x_train = np.arange(len(train_data))
    slope, intercept = np.polyfit(x_train, train_data, 1)
    x_full = np.arange(total_len)
    trend_full = slope * x_full + intercept
    return trend_full[:len(train_data)], trend_full[len(train_data):]


def compute_metrics(y_true, y_pred, train_data, m=1):
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    
    train_diff = train_data[m:] - train_data[:-m] if len(train_data) > m else train_data[1:] - train_data[:-1]
    scale = np.mean(np.abs(train_diff)) + 1e-12
    mase = float(mae / scale)
    return {"mase": mase, "rmse": rmse, "mae": mae}


def main():
    parser = argparse.ArgumentParser(description="Run Pure-Residual Hybrid Benchmarks")
    parser.add_argument("--n_lags", type=int, default=2)
    parser.add_argument("--train_fraction", type=float, default=0.80)
    args = parser.parse_args()

    if not SELECTED_CSV.exists():
        print(f"Error: {SELECTED_CSV} not found.")
        return

    selected = pd.read_csv(SELECTED_CSV)
    results = []

    print("="*60)
    print("Running Pure-Residual Hybrid Benchmarks...")
    print("="*60)

    for _, row in selected.iterrows():
        dataset = str(row["series_id"])
        freq = str(row["frequency"]).lower()
        season_length = 12 if "month" in freq else (4 if "quarter" in freq else 1)
        
        path = ROOT / str(row["file_path"])
        if not path.exists():
            continue
            
        df = pd.read_csv(path)
        col = "revenue_million_yer" if "revenue_million_yer" in df.columns else "value"
        values = df[col].dropna().to_numpy(dtype=float)

        n_train = int(len(values) * args.train_fraction)
        train_data = values[:n_train]
        test_data = values[n_train:]
        horizon = len(test_data)

        # 1. استخراج الاتجاه الخطي للسلسلة كاملة
        train_trend, test_trend = get_linear_trend(train_data, len(values))

        # 2. حساب البواقي الصافية لمرحلة التدريب
        train_residuals = train_data - train_trend

        # 3. تدريب نموذج NFPSO على البواقي حصرياً
        model = DynamicPCNFPSO(
            n_lags=args.n_lags,
            validation_size=max(2, int(len(train_data) * 0.15)),
            swarm_size=15,
            max_iter=40,
        )
        
        start_time = time.time()
        model.fit(train_residuals)
        
        # التنبؤ التراجعي للبواقي حصرياً في مساحة البواقي
        test_residuals_true = test_data - test_trend
        predicted_residuals = model.forecast(test_residuals_true)
        runtime = time.time() - start_time

        # 4. التنبؤ النهائي = الاتجاه الخطي للاختبار + البواقي المتنبأ بها
        hybrid_predictions = test_trend + predicted_residuals

        metrics = compute_metrics(test_data, hybrid_predictions, train_data, m=season_length)

        results.append({
            "dataset": dataset,
            "model": "projected_constricted_nfpso",
            "seed": 42,
            "runtime_seconds": runtime,
            "mase": metrics["mase"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"]
        })
        print(f"Dataset: {dataset:<20} | Pure-Residual Hybrid MASE: {metrics['mase']:.4f}")

    res_df = pd.DataFrame(results)
    out_metrics_path = OUT_DIR / "real_short_series_metrics_by_seed.csv"
    res_df.to_csv(out_metrics_path, index=False)
    print("="*60)
    print(f"Saved Pure-Residual metrics to {out_metrics_path}")
    print(f"Final Hybrid Mean MASE: {res_df['mase'].mean():.4f}")
    print("="*60)


if __name__ == "__main__":
    main()