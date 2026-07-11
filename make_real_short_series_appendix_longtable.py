from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "real_short_series"
METRICS_PATH = OUT_DIR / "real_short_series_metrics.csv"
OUT_PATH = OUT_DIR / "latex" / "table_real_short_series_metrics_longtable.tex"

MODEL_LABELS = {
    "projected_constricted_nfpso": "PC-NFPSO",
    "naive_1": "Naive",
    "seasonal_naive": "Seasonal naive",
    "drift": "Drift",
    "arima": "ARIMA",
    "ets": "ETS",
    "theta": "Theta",
    "svr": "SVR",
    "xgboost": "XGBoost",
}

MODEL_ORDER = [
    "projected_constricted_nfpso",
    "ets",
    "arima",
    "drift",
    "naive_1",
    "seasonal_naive",
    "theta",
    "xgboost",
    "svr",
]

def esc(x):
    return str(x).replace("_", "\\_")

def fmt(x):
    return f"{float(x):.3f}"

def main():
    df = pd.read_csv(METRICS_PATH)
    df = df[df["status"].fillna("ok").astype(str).str.lower().eq("ok")].copy()
    df["model_sort"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(999)
    df = df.sort_values(["dataset", "model_sort"])

    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{esc(r['dataset'])} & {MODEL_LABELS.get(r['model'], r['model'])} & "
            f"{fmt(r['rmse'])} & {fmt(r['mae'])} & "
            f"{fmt(r['smape_percent'])} & {fmt(r['mase'])} \\\\"
        )

    body = "\n".join(rows)

    tex = rf"""\scriptsize
\begin{{longtable}}{{llrrrr}}
\caption{{Real short-series forecasting results. Lower values are better. PC-NFPSO values are averaged across optimizer seeds.}}
\label{{tab:real_short_series_metrics}}\\
\hline
Series & Model & RMSE & MAE & sMAPE (\%) & MASE \\
\hline
\endfirsthead
\hline
Series & Model & RMSE & MAE & sMAPE (\%) & MASE \\
\hline
\endhead
{body}
\hline
\end{{longtable}}
\normalsize
"""
    OUT_PATH.write_text(tex, encoding="utf-8")
    print(f"Wrote {OUT_PATH}")

if __name__ == "__main__":
    main()