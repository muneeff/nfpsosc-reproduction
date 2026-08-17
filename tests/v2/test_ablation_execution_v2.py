from nfpsosc.v2.final_runner import build_final_tasks
from nfpsosc.v2.ablation_runner import ABLATION_VARIANTS
from nfpsosc.v2.ablation_execution import execute_ablation_variant


def test_ablation_single_final_task_nf_base():

    tasks = build_final_tasks()
    task = tasks[0]

    result = execute_ablation_variant(
        ABLATION_VARIANTS["NF_BASE"],
        task,
        project_root=".",
        radius=0.5,
        alpha=0.01,
        optimizer_seed=1,
    )

    print("NF_BASE FAILURE:", result.failure)

    assert result.status == "success"
    assert result.metrics is not None


def test_ablation_single_final_task_pc_nfpso():

    tasks = build_final_tasks()
    task = tasks[0]

    result = execute_ablation_variant(
        ABLATION_VARIANTS["PC_NFPSO"],
        task,
        project_root=".",
        radius=0.5,
        alpha=0.01,
        optimizer_seed=1,
    )

    assert result.status == "success"
    assert result.metrics is not None
    assert result.dynamics == "constricted"
    assert result.boundary == "project"