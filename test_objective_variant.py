from nfpsosc.v2.final_runner import build_final_tasks
from nfpsosc.v2.ablation_runner import ABLATION_VARIANTS
from nfpsosc.v2.final_execution import load_final_task_series
from nfpsosc.v2.ablation_runner import fit_ablation_variant
from nfpsosc.v2.objective import ObjectiveWeights


tasks = build_final_tasks()[:100]

variant = ABLATION_VARIANTS["PC_NFPSO"]

weights = ObjectiveWeights(
    fit_rmse=1.0,
    validation_rmse=1.0,
    sensitivity=0.0,
    low_coverage=0.0,
)

rmses = []

for i, task in enumerate(tasks):

    train, test = load_final_task_series(
        task,
        project_root=".",
    )

    fitted = fit_ablation_variant(
        variant,
        train,
        n_lags=task.n_lags,
        validation_size=len(test),
        radius=0.5,
        alpha=0.01,
        optimizer_seed=1,
    )

    from nfpsosc.v2.pc_nfpso import forecast_pc_nfpso_v2

    pred = forecast_pc_nfpso_v2(
        fitted,
        test,
    )

    from nfpsosc.v2.final_execution import compute_final_metrics

    metrics = compute_final_metrics(
        train,
        test,
        pred,
        seasonal_period=task.seasonal_period,
    )

    rmses.append(metrics.rmse)

    print(
        i + 1,
        metrics.rmse,
    )


print("Mean RMSE:", sum(rmses) / len(rmses))