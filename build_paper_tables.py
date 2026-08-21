from pathlib import Path
import pandas as pd
import numpy as np

SYNTHETIC_CSV = Path("outputs/q1_minimal/synthetic_benchmarks/metrics.csv")
REAL_CSV_OLD = Path("outputs/real_short_series/real_short_series_metrics.csv")
REAL_CSV_NEW = Path("outputs/real_short_series/real_short_series_metrics_by_seed.csv")
OUT = Path("paper_tables")
OUT.mkdir(exist_ok=True)

def process_and_save(df, name):
    df.columns = [c.upper() for c in df.columns]
    df["VARIANT"] = df.get("MODEL", df.get("VARIANT", "Unknown"))
    
    metrics = [m for m in ["MASE", "RMSE", "MAE", "SMAPE"] if m in df.columns]
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors='coerce')
        
    summary = df.groupby("VARIANT").agg({m: ["mean", "std"] for m in metrics})
    summary.columns = ['_'.join(c).strip() for c in summary.columns]
    summary = summary.reset_index()
    
    sort_col = "MASE_mean" if "MASE_mean" in summary.columns else "RMSE_mean"
    summary = summary.sort_values(sort_col)
    
    summary.to_csv(OUT / f"{name}_summary.csv", index=False)
    print(f"\n================== {name.upper()} SUMMARY ==================")
    print(summary.to_string(index=False))
    return summary

if SYNTHETIC_CSV.exists():
    df_syn = pd.read_csv(SYNTHETIC_CSV)
    process_and_save(df_syn, "synthetic_benchmarks")

if REAL_CSV_OLD.exists() and REAL_CSV_NEW.exists():
    df_old = pd.read_csv(REAL_CSV_OLD)
    df_new = pd.read_csv(REAL_CSV_NEW)
    
    # استبعاد النموذج القديم من القائمة القديمة ودمج النتائج الجديدة المحدثة
    df_old = df_old[df_old["model"] != "projected_constricted_nfpso"]
    df_real = pd.concat([df_old, df_new], ignore_index=True)
    
    process_and_save(df_real, "real_short_series")