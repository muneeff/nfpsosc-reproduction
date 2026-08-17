from concurrent.futures import ProcessPoolExecutor, as_completed

from .final_runner import build_final_tasks
from .ablation_full_runner import run_ablation_task


def _run_one_task(task):

    return run_ablation_task(
        task,
        project_root=".",
    )


def run_parallel_ablation(
    *,
    workers: int = 4,
):

    tasks = build_final_tasks()

    results = []

    with ProcessPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = [
            executor.submit(
                _run_one_task,
                task,
            )
            for task in tasks
        ]

        for future in as_completed(futures):

            results.extend(
                future.result()
            )

    return tuple(results)

def run_parallel_subset(
    n_tasks: int = 100,
    *,
    workers: int = 4,
):

    tasks = build_final_tasks()[:n_tasks]

    results = []

    with ProcessPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = [
            executor.submit(
                _run_one_task,
                task,
            )
            for task in tasks
        ]

        for future in as_completed(futures):
            results.extend(
                future.result()
            )

    return tuple(results)

__all__ = [
    "run_parallel_ablation",
    "run_parallel_subset",
]