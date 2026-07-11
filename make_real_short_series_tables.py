from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "real_short_series"
LATEX_DIR = OUT_DIR / "latex"

SELECTED_PATH = OUT_DIR / "selected_real_series.csv"
METRICS_PATH = OUT_DIR / "real_short_series_metrics.csv"

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

METRICS = ["rmse", "mae", "smape_percent", "mase"]
METRIC_LABELS = {
    "rmse": "RMSE",
    "mae": "MAE",
    "smape_percent": "sMAPE (\\%)",
    "mase": "MASE",
}


def fmt(x):
    if pd.isna(x):
        return "--"
    return f"{float(x):.3f}"


def esc(s):
    return str(s).replace("_", "\\_")


def model_label(model):
    return MODEL_LABELS.get(str(model), str(model))


def write(path, text):
    path.write_text(text, encoding="utf-8")
    print(f"[DONE] {path}")


def make_metadata_table(selected):
    rows = []
    for _, r in selected.iterrows():
        rows.append(
            f"{esc(r['series_id'])} & {esc(r['source'])} & {esc(r['frequency'])} & "
            f"{int(r['length'])} & {esc(r['selection_seed'])} \\\\"
        )

    body = "\n".join(rows)

    tex = rf"""\begin{{table*}}[!t]
\centering
\small
\caption{{Real short-series case studies used for external evaluation.}}
\label{{tab:real_short_series_metadata}}
\begin{{tabular}}{{lllrl}}
\hline
Series & Source & Frequency & Length & Selection seed \\
\hline
{body}
\hline
\end{{tabular}}
\end{{table*}}
"""
    write(LATEX_DIR / "table_real_short_series_metadata.tex", tex)


def make_best_models_table(metrics):
    rows = []

    for dataset, g in metrics.groupby("dataset", sort=False):
        cells = [esc(dataset)]
        for metric in METRICS:
            vals = pd.to_numeric(g[metric], errors="coerce")
            min_val = vals.min()
            best = g.loc[vals == min_val, "model"].tolist()
            best_names = "/".join(model_label(m) for m in best)
            cells.append(f"{best_names} ({fmt(min_val)})")
        rows.append(" & ".join(cells) + r" \\")

    body = "\n".join(rows)

    tex = rf"""\begin{{table*}}[!t]
\centering
\scriptsize
\caption{{Best-performing model for each real short-series case study. Lower values are better.}}
\label{{tab:real_short_series_best_models}}
\begin{{tabular}}{{lcccc}}
\hline
Series & RMSE & MAE & sMAPE (\%) & MASE \\
\hline
{body}
\hline
\end{{tabular}}
\end{{table*}}
"""
    write(LATEX_DIR / "table_real_short_series_best_models.tex", tex)


def make_full_metrics_table(metrics):
    df = metrics.copy()
    df["model_sort"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(999)
    df = df.sort_values(["dataset", "model_sort"])

    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{esc(r['dataset'])} & {model_label(r['model'])} & "
            f"{fmt(r['rmse'])} & {fmt(r['mae'])} & "
            f"{fmt(r['smape_percent'])} & {fmt(r['mase'])} \\\\"
        )

    body = "\n".join(rows)

    tex = rf"""\begin{{table*}}[!t]
\centering
\scriptsize
\caption{{Real short-series forecasting results. Lower values are better. PC-NFPSO values are averaged across optimizer seeds.}}
\label{{tab:real_short_series_metrics}}
\begin{{tabular}}{{llcccc}}
\hline
Series & Model & RMSE & MAE & sMAPE (\%) & MASE \\
\hline
{body}
\hline
\end{{tabular}}
\end{{table*}}
"""
    write(LATEX_DIR / "table_real_short_series_metrics.tex", tex)


def main():
    if not SELECTED_PATH.exists():
        raise FileNotFoundError(f"Missing {SELECTED_PATH}")
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing {METRICS_PATH}")

    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(SELECTED_PATH)
    metrics = pd.read_csv(METRICS_PATH)

    ok = metrics["status"].fillna("ok").astype(str).str.lower().eq("ok")
    metrics = metrics[ok].copy()

    missing = [m for m in METRICS if m not in metrics.columns]
    if missing:
        raise RuntimeError(f"Missing metric columns: {missing}")

    make_metadata_table(selected)
    make_best_models_table(metrics)
    make_full_metrics_table(metrics)

    print("[DONE] Generated all real short-series LaTeX tables.")


if __name__ == "__main__":
    main()