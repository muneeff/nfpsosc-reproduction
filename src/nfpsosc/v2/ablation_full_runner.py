from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .final_runner import (
    build_final_tasks,
    FinalRunTask,
)

from .ablation_runner import (
    ABLATION_VARIANTS,
    AblationVariant,
)

from .ablation_execution import (
    execute_ablation_variant,
    AblationRunResult,
)
from .ablation_workspace import (
    ensure_ablation_workspace,
    save_ablation_result,
    load_existing_results,
)

@dataclass(frozen=True)
class AblationTaskResult:
    task_id: str
    variant: str
    result: AblationRunResult



def build_ablation_variants() -> tuple[AblationVariant, ...]:
    return tuple(
        ABLATION_VARIANTS.values()
    )



def run_ablation_task(
    task: FinalRunTask,
    *,
    project_root: str = ".",
    radius: float = 0.5,
    alpha: float = 0.01,
    optimizer_seed: int = 1,
) -> tuple[AblationTaskResult, ...]:

    results = []

    for variant in build_ablation_variants():

        result = execute_ablation_variant(
            variant,
            task,
            project_root=project_root,
            radius=radius,
            alpha=alpha,
            optimizer_seed=optimizer_seed,
        )

        results.append(
            AblationTaskResult(
                task_id=task.run_id,
                variant=variant.name,
                result=result,
            )
        )

    return tuple(results)



def run_full_ablation(
    *,
    project_root: str = ".",
    workspace: str = "outputs/v2/ablation_full",
   
):

    ensure_ablation_workspace(workspace)

    existing = load_existing_results(workspace)

    tasks = build_final_tasks()

    all_results = []

    for task in tasks:

        for item in run_ablation_task(
            task,
            project_root=project_root,
        ):

            key = (
                f"{task.run_id}__{item.variant}"
            )

            if key in existing:
                continue

            save_ablation_result(
                workspace,
                key,
                {
                    "task_id": item.task_id,
                    "variant": item.variant,
                    "status": item.result.status,
                    "metrics": item.result.metrics,
                                "best_objective": item.result.best_objective,
                    "failure": item.result.failure,
                },
            )

            all_results.append(item)

    return tuple(all_results)
def run_small_ablation(
    n_tasks=10,
    *,
    project_root=".",
):

    tasks = build_final_tasks()[:n_tasks]

    results = []

    for task in tasks:
        results.extend(
            run_ablation_task(
                task,
                project_root=project_root,
            )
        )

    return tuple(results)

__all__ = [
    "AblationTaskResult",
    "build_ablation_variants",
    "run_ablation_task",
    "run_full_ablation",
]