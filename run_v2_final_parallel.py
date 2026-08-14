from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path("src").resolve()))

from nfpsosc.v2.final_parallel import (
    DEFAULT_WORKERS,
    FinalParallelError,
    run_pending_final_tasks_parallel,
)


def _progress(done: int, total: int, result) -> None:
    task = result.task
    if task.domain == "synthetic":
        identity = (
            f"{task.generator} dgp={task.dgp_seed} noise={task.noise_level} "
            f"method={task.method}"
        )
        if task.optimizer_seed is not None:
            identity += f" opt={task.optimizer_seed}"
    else:
        identity = (
            f"{task.source}/{task.frequency}/{task.series_id} "
            f"method={task.method}"
        )
        if task.optimizer_seed is not None:
            identity += f" opt={task.optimizer_seed}"
    print(
        f"[{done}/{total}] {result.status.upper()} "
        f"{identity} run={task.run_id[:12]}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen V2 Final external benchmark with bounded spawn-based "
            "parallelism. Run once with --max-runs 0 before creating the execution "
            "freeze tag and before any Final outcome."
        )
    )
    parser.add_argument(
        "--workspace",
        default=r"outputs\v2\final_external",
        help="Frozen Final external workspace.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Worker-process count (default: 4).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help=(
            "Maximum pending runs for this call. Use 0 for the mandatory pre-outcome "
            "execution-workspace freeze."
        ),
    )
    args = parser.parse_args()

    try:
        summary = run_pending_final_tasks_parallel(
            args.workspace,
            project_root=".",
            workers=args.workers,
            max_runs=args.max_runs,
            progress_callback=_progress,
        )
    except FinalParallelError as exc:
        print(f"FINAL_PARALLEL_ERROR: {exc}", file=sys.stderr)
        return 2

    print("WORKERS=", summary.workers)
    print("MAX_IN_FLIGHT=", summary.max_in_flight)
    print("SELECTED_THIS_CALL=", summary.selected_count)
    print("COMPLETED_THIS_CALL=", summary.completed_this_call)
    print("SUCCESSFUL_THIS_CALL=", summary.successful_this_call)
    print("FAILED_THIS_CALL=", summary.failed_this_call)
    print("COMPLETED_TOTAL=", summary.completed_total)
    print("SUCCESSFUL_TOTAL=", summary.successful_total)
    print("FAILED_TOTAL=", summary.failed_total)
    print("PENDING_TOTAL=", summary.pending_total)
    print("WALL_SECONDS=", f"{summary.wall_seconds:.2f}")
    print("STOPPED_AFTER_FAILURE=", summary.stopped_after_failure)

    if summary.failed_this_call:
        print("FINAL_PARALLEL_BATCH_STOPPED_ON_FAILURE")
        return 2
    if args.max_runs == 0:
        print("NO_FINAL_OUTCOMES_GENERATED= True")
        print("3F3B_FINAL_EXECUTION_WORKSPACE_FREEZE_PREP_PASS")
    else:
        print("FINAL_PARALLEL_BATCH_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
