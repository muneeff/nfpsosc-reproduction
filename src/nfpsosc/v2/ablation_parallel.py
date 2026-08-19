from concurrent.futures import ProcessPoolExecutor, as_completed

from .final_runner import build_final_tasks
from .ablation_full_runner import run_ablation_task
from .ablation_workspace import (
    ensure_ablation_workspace,
    save_ablation_result,
)


def _run_one_task(task):

    return run_ablation_task(
        task,
        project_root=".",
    )


def _save_task_results(
    results,
    *,
    workspace: str,
):
    for item in results:

        result = item.result

        payload = {
            "task_id": item.task_id,
            "variant": item.variant,
            "status": result.status,
            "dynamics": result.dynamics,
            "boundary": result.boundary,
            "metrics": result.metrics,
            "n_rules": result.n_rules,
            "best_objective": result.best_objective,
            "failure": result.failure,
        }

        run_id = f"{item.task_id}_{item.variant}"

        save_ablation_result(
            workspace,
            run_id,
            payload,
        )


def run_parallel_ablation(
    *,
    workers: int = 4,
    workspace: str = "outputs/v2/ablation_full",
):

    ensure_ablation_workspace(workspace)

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

        completed = 0

        for future in as_completed(futures):

            task_results = future.result()

            _save_task_results(
                task_results,
                workspace=workspace,
            )

            results.extend(task_results)

            completed += 1

            if completed % 50 == 0:
                print(
                    f"Completed tasks: {completed}/{len(tasks)}"
                )

    return tuple(results)


def run_parallel_subset(
    n_tasks: int = 100,
    *,
    workers: int = 4,
    workspace: str = "outputs/v2/ablation_test",
):

    ensure_ablation_workspace(workspace)

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

            task_results = future.result()

            _save_task_results(
                task_results,
                workspace=workspace,
            )

            results.extend(task_results)

    return tuple(results)


__all__ = [
    "run_parallel_ablation",
    "run_parallel_subset",
]