from pathlib import Path
import pandas as pd


def fmt_num(x, digits=3):
    if pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


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


def main():
    input_path = Path(r"outputs\q1_minimal\synthetic_nfpso_experiments\metrics.csv")
    out_dir = Path(r"outputs\q1_minimal\paper_tables\latex")
    csv_dir = Path(r"outputs\q1_minimal\paper_tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run synthetic NFPSO sensitivity first."
        )

    df = pd.read_csv(input_path)
    if "model" in df.columns:
        df = df[df["model"] == "projected_constricted_nfpso"].copy()
    if "status" in df.columns:
        df = df[df["status"].fillna("ok").astype(str).str.lower().eq("ok")].copy()

    required = {"base_name", "sc_radius", "rmse"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["sc_radius"] = pd.to_numeric(df["sc_radius"], errors="coerce")
    df = df.dropna(subset=["sc_radius", "rmse"])

    if df.empty:
        raise ValueError("No valid projected_constricted_nfpso sensitivity rows found.")

    pivot = df.pivot_table(
        index="base_name",
        columns="sc_radius",
        values="rmse",
        aggfunc="mean",
    ).sort_index(axis=1)

    overall = df.groupby("sc_radius")["rmse"].agg(["mean", "median", "std", "count"]).sort_index()

    pivot.to_csv(csv_dir / "radius_sensitivity_by_generator.csv")
    overall.to_csv(csv_dir / "radius_sensitivity_overall.csv")

    radius_cols = list(pivot.columns)
    headers = ["Generator"] + [fmt_num(r, 2) for r in radius_cols] + ["Best radius", "Best RMSE"]
    rows = []

    for base_name, row in pivot.iterrows():
        best_radius = row.idxmin()
        best_rmse = row.min()
        rows.append(
            [base_name]
            + [fmt_num(row.get(r), 3) for r in radius_cols]
            + [fmt_num(best_radius, 2), fmt_num(best_rmse, 3)]
        )

    # Add overall row
    best_overall = overall["mean"].idxmin()
    rows.append(
        ["Overall mean"]
        + [fmt_num(overall.loc[r, "mean"], 3) for r in radius_cols]
        + [fmt_num(best_overall, 2), fmt_num(overall.loc[best_overall, "mean"], 3)]
    )

    body = simple_latex_tabular(headers, rows, "l" + "c" * (len(headers) - 1))
    caption = (
        "Sensitivity of PC-NFPSO to the subtractive-clustering radius. "
        "Entries are mean RMSE values; lower is better."
    )
    latex = (
        "\\begin{table*}[!t]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{caption}}}\n"
        "\\label{tab:radius_sensitivity}\n"
        + body +
        "\\end{table*}\n"
    )

    tex_path = out_dir / "table_radius_sensitivity.tex"
    tex_path.write_text(latex, encoding="utf-8")

    claims = []
    claims.append("Radius sensitivity summary")
    claims.append("==========================")
    claims.append("")
    claims.append(f"- Best overall radius by mean RMSE: {fmt_num(best_overall, 2)}.")
    claims.append(f"- Best overall mean RMSE: {fmt_num(overall.loc[best_overall, 'mean'], 3)}.")
    claims.append("")
    claims.append("Best radius by generator:")
    for base_name, row in pivot.iterrows():
        claims.append(f"- {base_name}: radius={fmt_num(row.idxmin(), 2)}, RMSE={fmt_num(row.min(), 3)}")

    (out_dir / "radius_sensitivity_claims.txt").write_text("\n".join(claims) + "\n", encoding="utf-8")

    print("[DONE] Radius sensitivity tables written to:")
    print(csv_dir / "radius_sensitivity_by_generator.csv")
    print(csv_dir / "radius_sensitivity_overall.csv")
    print(tex_path)
    print(out_dir / "radius_sensitivity_claims.txt")
    print("[NOTE] outputs/ is ignored by Git and should not be committed.")


if __name__ == "__main__":
    main()
