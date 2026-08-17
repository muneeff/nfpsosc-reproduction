from dataclasses import dataclass, asdict

import numpy as np

from .ablation_runner import (
    AblationVariant,
    fit_ablation_variant,
)

from .final_execution import (
    load_final_task_series,
    compute_final_metrics,
)


def forecast_nf_base_v2(
    fitted,
    test_actuals,
):
    """
    Rolling one-step forecast for frozen NF baseline.

    Same evaluation protocol as PC-NFPSO:
    - fixed model parameters
    - observed history update
    - no refitting during forecasting
    """

    actuals = np.asarray(test_actuals, dtype=float)

    training = fitted["training"]
    model = fitted["model"]

    history = training.raw_pretest.astype(float, copy=True)

    L = int(training.n_lags)

    if len(history) < L:
        raise ValueError(
            "Stored pre-test history is shorter than n_lags."
        )

    predictions = np.empty(len(actuals), dtype=float)

    for i, observed in enumerate(actuals):

        lag_raw = history[-L:]

        lag_scaled = (
            training.scaler
            .transform(lag_raw)
            .reshape(1, -1)
        )

        pred_scaled = float(
            model.predict(lag_scaled)[0]
        )

        pred_raw = float(
            training.scaler.inverse_transform(
                np.asarray([pred_scaled])
            )[0]
        )

        if not np.isfinite(pred_raw):
            raise ValueError(
                "NF_BASE produced non-finite forecast."
            )

        predictions[i] = pred_raw

        # observed history update without refitting
        history = np.append(
            history,
            float(observed)
        )

    return predictions


@dataclass(frozen=True)
class AblationRunResult:
    variant: str
    status: str
    dynamics: str | None
    boundary: str | None
    metrics: dict[str, float] | None
    n_rules: int | None = None
    best_objective: float | None = None
    failure: str | None = None


class AblationExecutionError(RuntimeError):
    pass


def execute_ablation_variant(
    variant: AblationVariant,
    task,
    *,
    project_root: str = ".",
    radius: float,
    alpha: float,
    optimizer_seed: int,
) -> AblationRunResult:
    """
    Execute one frozen ablation variant under the Final
    leakage-aware benchmarking protocol.

    Data loading and metrics are identical to Final execution.
    Only the internal optimization variant changes.
    """

    try:

        train, test = load_final_task_series(
            task,
            project_root=project_root,
        )

        fitted = fit_ablation_variant(
            variant,
            train,
            n_lags=task.n_lags,
            validation_size=len(test),
            radius=radius,
            alpha=alpha,
            optimizer_seed=optimizer_seed,
        )

    except Exception as exc:

        return AblationRunResult(
            variant=variant.name,
            status="failed",
            dynamics=variant.dynamics,
            boundary=variant.boundary,
            metrics=None,
            failure=str(exc),
        )


    try:

        if variant.uses_pso:

            from .pc_nfpso import forecast_pc_nfpso_v2

            prediction = forecast_pc_nfpso_v2(
                fitted,
                test,
            )

            n_rules = int(
                fitted.final_model.n_rules
            )

            best_objective = float(
                fitted.optimizer_result.best_cost
            )


        else:

            prediction = forecast_nf_base_v2(
                fitted,
                test,
            )

            model = fitted["model"]

            n_rules = int(
                model.n_rules
            )

            best_objective = None


        metrics = compute_final_metrics(
            train,
            test,
            prediction,
            seasonal_period=task.seasonal_period,
        )


    except Exception as exc:

        return AblationRunResult(
            variant=variant.name,
            status="failed",
            dynamics=variant.dynamics,
            boundary=variant.boundary,
            metrics=None,
            failure=str(exc),
        )


    return AblationRunResult(
        variant=variant.name,
        status="success",
        dynamics=variant.dynamics,
        boundary=variant.boundary,
        metrics=asdict(metrics),
        n_rules=n_rules,
        best_objective=best_objective,
        failure=None,
    )


__all__ = [
    "AblationRunResult",
    "AblationExecutionError",
    "forecast_nf_base_v2",
    "execute_ablation_variant",
]