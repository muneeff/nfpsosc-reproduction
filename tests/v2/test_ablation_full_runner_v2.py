from nfpsosc.v2.final_runner import build_final_tasks
from nfpsosc.v2.ablation_full_runner import run_ablation_task


def test_full_ablation_single_task():

    task = build_final_tasks()[0]

    results = run_ablation_task(
        task,
        project_root=".",
    )

    assert len(results) == 8

    for item in results:
        print(
            item.variant,
            item.result.status,
            item.result.failure,
        )

        assert item.result.status == "success"
        assert item.result.metrics is not None