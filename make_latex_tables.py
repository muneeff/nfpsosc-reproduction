from pathlib import Path
import pandas as pd


MODEL_LABELS = {
    "projected_constricted_nfpso": "PC-NFPSO",
    "ets": "ETS",
    "svr": "SVR",
    "xgboost": "XGBoost",
    "arima": "ARIMA",
    "seasonal_naive": "Seasonal naive",
    "naive_1": "Naive",
    "theta": "Theta",
    "drift": "Drift",
}

METRIC_LABELS = {
    "rmse": "RMSE",
    "mae": "MAE",
    "smape_percent": "sMAPE (\\%)",
    "mase": "MASE",
}

METRIC_ORDER = ["rmse", "mae", "smape_percent", "mase"]
MODEL_ORDER = [
    "projected_constricted_nfpso",
    "ets",
    "svr",
    "xgboost",
    "arima",
    "seasonal_naive",
    "naive_1",
    "theta",
    "drift",
]


def fmt_num(x, digits=3):
    if pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def fmt_p(x):
    if pd.isna(x):
        return "--"
    x = float(x)
    if x < 1e-3:
        return f"{x:.2e}"
    return f"{x:.3f}"


def tex_escape_text(value):
    s = str(value)
    if "\\" in s or "$" in s:
        return s
    return (
        s.replace("&", "\\&")
         .replace("%", "\\%")
         .replace("_", "\\_")
         .replace("#", "\\#")
    )


def simple_latex_tabular(headers, rows, col_format):
    lines = []
    lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    lines.append("\\hline")
    lines.append(" & ".join(headers) + r" \\")
    lines.append("\\hline")
    for row in rows:
        lines.append(" & ".join(tex_escape_text(x) for x in row) + r" \\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def make_performance_table(summary, out_dir):
    headers = ["Model"]
    for metric in METRIC_ORDER:
        headers.extend([METRIC_LABELS[metric], "Rank"])

    rows = []
    for model in MODEL_ORDER:
        subset = summary[summary["model"] == model]
        if subset.empty:
            continue
        row = [MODEL_LABELS.get(model, model)]
        for metric in METRIC_ORDER:
            m = subset[subset["metric"] == metric]
            if m.empty:
                row.extend(["--", "--"])
            else:
                item = m.iloc[0]
                row.extend([fmt_num(item["mean"]), fmt_num(item["rank_mean"], 2)])
        rows.append(row)

    body = simple_latex_tabular(headers, rows, "l" + "cc" * len(METRIC_ORDER))
    caption = (
        "Balanced synthetic benchmark summary over 100 conditions. "
        "Lower metric values and lower average ranks are better."
    )
    latex = (
        "\\begin{table*}[!t]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{caption}}}\n"
        "\\label{tab:synthetic_metric_summary}\n"
        + body +
        "\\end{table*}\n"
    )

    (out_dir / "table_synthetic_metric_summary.tex").write_text(latex, encoding="utf-8")
    pd.DataFrame(rows, columns=headers).to_csv(
        out_dir / "table_synthetic_metric_summary_for_paper.csv", index=False
    )


def make_pairwise_table(comps, out_dir):
    headers = [
        "Metric",
        "Comparator",
        "Direction",
        "$\\Delta$ mean",
        "95\\% CI",
        "$p_{Holm}$",
        "Sig.",
    ]

    rows = []
    model_sort = {m: i for i, m in enumerate(MODEL_ORDER)}

    for metric in METRIC_ORDER:
        subset = comps[comps["metric"] == metric].copy()
        subset["challenger_sort"] = subset["challenger_model"].map(model_sort).fillna(999)
        subset = subset.sort_values("challenger_sort")

        for _, r in subset.iterrows():
            direction = (
                "PC-NFPSO better"
                if r["direction_by_mean"] == "reference_better"
                else "Comparator better"
            )
            ci = f"[{fmt_num(r['ci_low'])}, {fmt_num(r['ci_high'])}]"
            rows.append(
                [
                    METRIC_LABELS.get(metric, metric),
                    MODEL_LABELS.get(r["challenger_model"], r["challenger_model"]),
                    direction,
                    fmt_num(r["mean_difference"]),
                    ci,
                    fmt_p(r["p_adjusted_holm"]),
                    "Yes" if bool(r["reject_alpha_holm"]) else "No",
                ]
            )

    body = simple_latex_tabular(headers, rows, "lllllll")
    caption = (
        "Pairwise Wilcoxon signed-rank comparisons against PC-NFPSO with Holm correction. "
        "Positive mean differences indicate lower error for PC-NFPSO."
    )
    latex = (
        "\\begin{table*}[!t]\n"
        "\\centering\n"
        "\\scriptsize\n"
        f"\\caption{{{caption}}}\n"
        "\\label{tab:synthetic_pairwise_holm}\n"
        + body +
        "\\end{table*}\n"
    )

    (out_dir / "table_synthetic_pairwise_holm.tex").write_text(latex, encoding="utf-8")
    pd.DataFrame(rows, columns=headers).to_csv(
        out_dir / "table_synthetic_pairwise_holm_for_paper.csv", index=False
    )


def make_short_claims(summary, comps, out_dir):
    lines = []
    lines.append("Synthetic benchmark claims supported by the current tables")
    lines.append("=" * 62)
    lines.append("")

    for metric in METRIC_ORDER:
        subset = summary[summary["metric"] == metric].sort_values("mean")
        best = subset.iloc[0]
        lines.append(
            f"- {METRIC_LABELS[metric]} best mean: "
            f"{MODEL_LABELS.get(best['model'], best['model'])} = {fmt_num(best['mean'])}."
        )

    lines.append("")
    lines.append("Pairwise Holm-corrected significance against PC-NFPSO:")
    for metric in METRIC_ORDER:
        subset = comps[comps["metric"] == metric]
        significant = subset[subset["reject_alpha_holm"] == True]["challenger_model"].tolist()
        nonsignificant = subset[subset["reject_alpha_holm"] == False]["challenger_model"].tolist()
        lines.append(
            f"- {METRIC_LABELS[metric]} significant vs: "
            + ", ".join(MODEL_LABELS.get(x, x) for x in significant)
        )
        lines.append(
            "  Not significant vs: "
            + (", ".join(MODEL_LABELS.get(x, x) for x in nonsignificant) if nonsignificant else "none")
        )

    (out_dir / "synthetic_benchmark_claims.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    in_dir = Path(r"outputs\q1_minimal\paper_tables")
    out_dir = in_dir / "latex"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = in_dir / "paper_table_synthetic_metric_summary.csv"
    comps_path = in_dir / "paper_table_synthetic_pairwise_holm.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    if not comps_path.exists():
        raise FileNotFoundError(f"Missing {comps_path}")

    summary = pd.read_csv(summary_path)
    comps = pd.read_csv(comps_path)

    make_performance_table(summary, out_dir)
    make_pairwise_table(comps, out_dir)
    make_short_claims(summary, comps, out_dir)

    print("[DONE] LaTeX tables written to:")
    print(out_dir / "table_synthetic_metric_summary.tex")
    print(out_dir / "table_synthetic_pairwise_holm.tex")
    print(out_dir / "synthetic_benchmark_claims.txt")
    print("[NOTE] This version does not require Jinja2.")
    print("[NOTE] outputs/ is ignored by Git and should not be committed.")


if __name__ == "__main__":
    main()
