from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nfpsosc.experiment import ExperimentConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat PSO runs across random seeds.")
    parser.add_argument("--protocol", choices=["legacy_intent","research_baseline"], default="legacy_intent")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11,22,33,44,55,66,77,88,99,111])
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--particles", type=int, default=25)
    args = parser.parse_args()

    rows = []
    base = ROOT / "outputs" / f"seed_study_{args.protocol}"
    for seed in args.seeds:
        cfg = ExperimentConfig(
            csv_path=str(ROOT / "data" / "yemen_tax_revenues_2002_2014.csv"),
            output_dir=str(base / f"seed_{seed}"),
            protocol=args.protocol,
            objective="error_std" if args.protocol == "legacy_intent" else "composite",
            sensitivity_weight=0.001,
            iterations=args.iterations,
            particles=args.particles,
            seed=seed,
        )
        summary = run_experiment(cfg)
        rows.append({
            "seed": seed,
            "best_objective": summary["diagnostics"]["best_objective"],
            "test_rmse": summary["test_metrics"]["rmse"],
            "test_mape_percent": summary["test_metrics"]["mape_percent"],
            "test_r2": summary["test_metrics"]["r2"],
            "rules": summary["diagnostics"]["n_rules"],
            "jacobian_max": summary["diagnostics"]["jacobian_norm_max"],
            "empirical_lipschitz_max": summary["diagnostics"]["empirical_lipschitz_max"],
            "min_activation_sum": summary["diagnostics"]["min_activation_sum"],
        })
    df = pd.DataFrame(rows)
    base.mkdir(parents=True, exist_ok=True)
    df.to_csv(base / "seed_study_summary.csv", index=False)
    print(df.to_string(index=False))
    print("\nAggregate:")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
