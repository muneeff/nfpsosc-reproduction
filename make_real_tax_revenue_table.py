from pathlib import Path
import pandas as pd


METRICS_PATH = Path("outputs/q1_minimal/real_data_benchmarks/metrics.csv")
OUT_DIR = Path("outputs/q1_minimal/paper_tables/latex")
OUT_TEX = OUT_DIR / "table_real_tax_revenue_metrics.tex"


DISPLAY_NAMES = {
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


def find_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols:
            return cols[name.lower()]
    return None


def fmt(x):
    if pd.isna(x):
        return "--"
    return f"{float(x):.3f}"


def main():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {METRICS_PATH}. Run real-data benchmarks first:\n"
            "set PYTHONPATH=src\n"
            "py run_real_data_benchmarks.py"
        )

    df = pd.read_csv(METRICS_PATH)

    model_col = find_col(df, ["model"])
    if model_col is None:
        raise ValueError(f"No model column found. Columns are: {list(df.columns)}")

    rmse_col = find_col(df, ["rmse", "RMSE"])
    mae_col = find_col(df, ["mae", "MAE"])
    smape_col = find_col(df, ["smape", "sMAPE", "smape_percent", "sMAPE (%)"])
    mase_col = find_col(df, ["mase", "MASE"])

    metric_cols = [c for c in [rmse_col, mae_col, smape_col, mase_col] if c is not None]
    if not metric_cols:
        raise ValueError(f"No metric columns found. Columns are: {list(df.columns)}")

    summary = df.groupby(model_col, as_index=False)[metric_cols].mean()
    summary["DisplayModel"] = summary[model_col].map(DISPLAY_NAMES).fillna(summary[model_col])

    if rmse_col is not None:
        summary = summary.sort_values(rmse_col, ascending=True)
    else:
        summary = summary.sort_values("DisplayModel")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Real tax-revenue case-study forecasting results. Lower values are better.}")
    lines.append(r"\label{tab:real_tax_revenue_metrics}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\hline")
    lines.append(r"Model & RMSE & MAE & sMAPE (\%) & MASE \\")
    lines.append(r"\hline")

    for _, row in summary.iterrows():
        model = row["DisplayModel"]
        rmse = fmt(row[rmse_col]) if rmse_col else "--"
        mae = fmt(row[mae_col]) if mae_col else "--"
        smape = fmt(row[smape_col]) if smape_col else "--"
        mase = fmt(row[mase_col]) if mase_col else "--"
        lines.append(f"{model} & {rmse} & {mae} & {smape} & {mase} \\\\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {OUT_TEX}")


if __name__ == "__main__":
    main()
