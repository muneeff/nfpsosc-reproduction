from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nfpsosc.data import chronological_split, create_lagged_data, load_revenue_series
from nfpsosc.metrics import regression_metrics
from nfpsosc.serialization import load_model_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Input-noise sensitivity study for a saved TSK model.")
    parser.add_argument("--experiment", default=str(ROOT / "outputs" / "legacy_intent"))
    parser.add_argument("--levels", nargs="+", type=float, default=[0.0,0.01,0.05,0.10,0.20])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    exp = Path(args.experiment)
    summary = json.loads((exp / "summary.json").read_text(encoding="utf-8"))
    model, scaler = load_model_npz(exp / "model.npz")
    csv_path = Path(summary["config"]["csv_path"])
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    raw = load_revenue_series(csv_path)["revenue_million_yer"].to_numpy(float)
    scaled = scaler.transform(raw)
    X, y = create_lagged_data(
        scaled,
        summary["config"]["n_lags"],
        summary["config"]["lag_order"],
    )
    _, _, Xtest, ytest = chronological_split(X, y, summary["config"]["train_fraction"])
    ytest_raw = scaler.inverse_transform(ytest)
    feature_scale = np.std(Xtest, axis=0, ddof=1)
    feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
    rng = np.random.default_rng(args.seed)

    rows = []
    for level in args.levels:
        for trial in range(args.trials):
            noise = rng.normal(size=Xtest.shape) * feature_scale * level
            pred = scaler.inverse_transform(model.predict(Xtest + noise))
            m = regression_metrics(ytest_raw, pred)
            clean_pred = model.predict(Xtest)
            perturbed_pred = model.predict(Xtest + noise)
            ratio = np.abs(perturbed_pred-clean_pred) / np.maximum(np.linalg.norm(noise,axis=1),1e-12)
            rows.append({
                "noise_level": level,
                "trial": trial,
                "rmse": m["rmse"],
                "mape_percent": m["mape_percent"],
                "sensitivity_mean": float(np.mean(ratio)) if level > 0 else 0.0,
                "sensitivity_max": float(np.max(ratio)) if level > 0 else 0.0,
            })
    df = pd.DataFrame(rows)
    df.to_csv(exp / "noise_sensitivity.csv", index=False)
    agg = df.groupby("noise_level", as_index=False).agg(
        rmse_mean=("rmse","mean"), rmse_std=("rmse","std"),
        mape_mean=("mape_percent","mean"),
        sensitivity_mean=("sensitivity_mean","mean"),
        sensitivity_max=("sensitivity_max","max"),
    )
    agg.to_csv(exp / "noise_sensitivity_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.errorbar(agg["noise_level"], agg["rmse_mean"], yerr=agg["rmse_std"], marker="o")
    ax.set_xlabel("Relative input-noise level")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Noise robustness")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(exp / "noise_robustness.png", dpi=180)
    plt.close(fig)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
