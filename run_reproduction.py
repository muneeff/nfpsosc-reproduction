from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nfpsosc.experiment import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce and audit the NF-PSO-SC Yemen revenue experiment."
    )
    parser.add_argument(
        "--csv",
        default=str(ROOT / "data" / "yemen_tax_revenues_2002_2014.csv"),
    )
    parser.add_argument(
        "--protocol",
        choices=["legacy_intent", "research_baseline"],
        default="legacy_intent",
    )
    parser.add_argument(
        "--objective",
        choices=["error_std", "rmse", "composite"],
        default=None,
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--particles", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--radius", type=float, default=0.55)
    parser.add_argument("--sensitivity-weight", type=float, default=0.001)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.iterations = 40
        args.particles = 12
    objective = args.objective
    if objective is None:
        objective = "error_std" if args.protocol == "legacy_intent" else "composite"
    output = ROOT / "outputs" / args.protocol
    config = ExperimentConfig(
        csv_path=args.csv,
        output_dir=str(output),
        protocol=args.protocol,
        objective=objective,
        sensitivity_weight=args.sensitivity_weight,
        seed=args.seed,
        iterations=args.iterations,
        particles=args.particles,
        sc_radius=args.radius,
    )
    summary = run_experiment(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nOutputs written to: {output}")


if __name__ == "__main__":
    main()
