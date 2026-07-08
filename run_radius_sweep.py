from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nfpsosc.clustering import initialize_tsk_from_sc
from nfpsosc.data import MinMaxScaler, chronological_split, create_lagged_data, load_revenue_series
from nfpsosc.diagnostics import model_diagnostics
from nfpsosc.metrics import regression_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Study SC radius, rule count, error, and sensitivity.")
    parser.add_argument("--radii", nargs="+", type=float, default=[0.2,0.3,0.4,0.5,0.55,0.6,0.7,0.8])
    parser.add_argument("--csv", default=str(ROOT / "data" / "yemen_tax_revenues_2002_2014.csv"))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "radius_sweep"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_revenue_series(args.csv)
    raw = df["revenue_million_yer"].to_numpy(float)
    scaler = MinMaxScaler.fit(raw, 0.1, 0.9)
    scaled = scaler.transform(raw)
    X, y = create_lagged_data(scaled, 5, "chronological")
    Xtr, ytr, Xte, yte = chronological_split(X, y, 0.85)
    yte_raw = scaler.inverse_transform(yte)

    rows = []
    for radius in args.radii:
        started = time.perf_counter()
        model, sc = initialize_tsk_from_sc(Xtr, ytr, radius=radius)
        pred = scaler.inverse_transform(model.predict(Xte))
        metrics = regression_metrics(yte_raw, pred)
        diag = model_diagnostics(model, Xte, seed=123)
        center_dist = np.inf
        if model.n_rules > 1:
            diff = sc.centers_normalized[:,None,:-1] - sc.centers_normalized[None,:,:-1]
            dist = np.linalg.norm(diff, axis=2)
            dist[dist == 0] = np.inf
            center_dist = float(np.min(dist))
        rows.append({
            "radius": radius,
            "rules": model.n_rules,
            "parameters": len(model.to_vector()),
            "test_rmse": metrics["rmse"],
            "test_mape_percent": metrics["mape_percent"],
            "jacobian_max": diag["jacobian_norm_max"],
            "empirical_lipschitz_max": diag["empirical_lipschitz_max"],
            "min_activation_sum": diag["min_activation_sum"],
            "minimum_center_separation_normalized": center_dist,
            "runtime_seconds": time.perf_counter() - started,
        })

    result = pd.DataFrame(rows)
    result.to_csv(out / "radius_sweep.csv", index=False)

    for y_col, title in [
        ("rules", "SC radius vs number of rules"),
        ("test_rmse", "SC radius vs test RMSE"),
        ("empirical_lipschitz_max", "SC radius vs empirical sensitivity"),
    ]:
        fig, ax = plt.subplots(figsize=(8,4.5))
        ax.plot(result["radius"], result[y_col], marker="o")
        ax.set_xlabel("SC radius")
        ax.set_ylabel(y_col.replace("_", " "))
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"{y_col}_vs_radius.png", dpi=180)
        plt.close(fig)

    print(result.to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
