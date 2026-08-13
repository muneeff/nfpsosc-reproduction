from __future__ import annotations

# Safe entry point for the frozen V2 development execution.
# This file intentionally imports only stdlib before configuring worker thread caps.

import argparse
import os
from pathlib import Path
import sys

THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
for _name in THREAD_ENV_NAMES:
    os.environ[_name] = "1"

sys.path.insert(0, str(Path("src").resolve()))

from nfpsosc.v2.development_parallel import (
    DEFAULT_WORKERS,
    run_pending_development_tasks_parallel,
)


def _progress(done, total, result):
    status = result.status.upper()
    print(
        f"[{done}/{total}] {status} "
        f"{result.task.generator} dgp={result.task.dgp_seed} "
        f"noise={result.task.noise_level} opt={result.task.optimizer_seed} "
        f"r={result.task.radius} alpha={result.task.alpha} "
        f"run={result.task.run_id[:12]}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        default=r"outputs\v2\development_selection",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    summary = run_pending_development_tasks_parallel(
        args.workspace,
        project_root=".",
        workers=args.workers,
        max_runs=args.max_runs,
        fail_fast=True,
        progress_callback=_progress,
    )
    print("WORKERS=", summary.workers)
    print("SELECTED_THIS_CALL=", summary.selected_count)
    print("COMPLETED_THIS_CALL=", summary.completed_this_call)
    print("SUCCESSFUL_THIS_CALL=", summary.successful_this_call)
    print("FAILED_THIS_CALL=", summary.failed_this_call)
    print("COMPLETED_TOTAL=", summary.completed_total)
    print("SUCCESSFUL_TOTAL=", summary.successful_total)
    print("FAILED_TOTAL=", summary.failed_total)
    print("PENDING_TOTAL=", summary.pending_total)
    print("WALL_SECONDS=", round(summary.wall_seconds, 3))
    print("STOPPED_AFTER_FAILURE=", summary.stopped_after_failure)

    if summary.failed_total:
        print("DEVELOPMENT_PARALLEL_STOPPED_ON_FAILURE")
        return 2
    print("DEVELOPMENT_PARALLEL_BATCH_PASS")
    return 0


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    raise SystemExit(main())
