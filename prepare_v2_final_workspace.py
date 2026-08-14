from __future__ import annotations

# Manifest-only entry point for V2 Final external benchmarking.
# No model import beyond the manifest layer; no fit/forecast/result execution.

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path("src").resolve()))

from nfpsosc.v2.final_runner import (
    FINAL_OUTCOME_EXECUTION_ENABLED,
    final_dry_run_summary,
    prepare_final_workspace,
    protocol_fingerprint,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the frozen V2 Final external task manifest only. "
            "This command cannot execute Final forecasts."
        )
    )
    parser.add_argument(
        "--workspace",
        default=r"outputs\v2\final_external",
    )
    args = parser.parse_args()

    if FINAL_OUTCOME_EXECUTION_ENABLED:
        raise RuntimeError("Manifest-only stage unexpectedly enables Final execution.")

    summary = final_dry_run_summary(project_root=".")
    manifest = prepare_final_workspace(
        args.workspace,
        project_root=".",
    )

    print("FINAL_OUTCOME_EXECUTION_ENABLED=", FINAL_OUTCOME_EXECUTION_ENABLED)
    print("CODE_COMMIT=", manifest.code_commit)
    print("PROTOCOL_FINGERPRINT=", manifest.protocol_fingerprint)
    print("TASKS_TOTAL=", summary.task_count)
    print("SYNTHETIC_TASKS=", summary.synthetic_task_count)
    print("SYNTHETIC_PC_TASKS=", summary.synthetic_pc_task_count)
    print("SYNTHETIC_BASELINE_TASKS=", summary.synthetic_baseline_task_count)
    print("REAL_TASKS=", summary.real_task_count)
    print("REAL_PC_TASKS=", summary.real_pc_task_count)
    print("REAL_BASELINE_TASKS=", summary.real_baseline_task_count)
    print("REAL_FINAL_SERIES=", summary.real_final_series_count)
    print("FINAL_DGP_SEEDS=", summary.final_dgp_seed_count)
    print("FINAL_OPTIMIZER_SEEDS=", summary.final_optimizer_seed_count)
    print("SELECTED_RADIUS=", summary.selected_radius)
    print("SELECTED_ALPHA=", summary.selected_alpha)
    print("FIRST_RUN_ID=", summary.first_run_id)
    print("LAST_RUN_ID=", summary.last_run_id)
    print("NO_FINAL_OUTCOMES_GENERATED= True")
    print("3F2_FINAL_MANIFEST_PREP_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
