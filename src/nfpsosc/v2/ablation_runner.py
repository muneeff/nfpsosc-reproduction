from dataclasses import dataclass

import numpy as np

from .objective import ObjectiveWeights
from .pc_nfpso import (
    prepare_pc_nfpso_training_data,
    fit_pc_nfpso_v2,
)
from .initialization import initialize_tsk_from_subtractive_clustering


def fit_nf_base_v2(
    raw_pretest: np.ndarray,
    *,
    n_lags: int,
    validation_size: int,
    radius: float,
    alpha: float,
):
    """
    Frozen NF baseline:
    subtractive clustering + ridge consequents.
    No antecedent PSO optimization.
    """

    training = prepare_pc_nfpso_training_data(
        raw_pretest,
        n_lags=n_lags,
        validation_size=validation_size,
    )

    initialization = initialize_tsk_from_subtractive_clustering(
        training.X_fit,
        training.y_fit,
        radius=float(radius),
    )

    model = initialization.model.with_antecedents(
        initialization.particle0
    )

    model.fit_consequents(
        training.X_pretest,
        training.y_pretest,
        alpha=float(alpha),
    )

    return {
        "training": training,
        "model": model,
        "radius": float(radius),
        "alpha": float(alpha),
    }


@dataclass(frozen=True)
class AblationVariant:
    name: str
    uses_pso: bool
    dynamics: str | None
    boundary: str | None
    objective_weights: ObjectiveWeights


BASE_OBJECTIVE = ObjectiveWeights(
    fit_rmse=0.5,
    validation_rmse=1.0,
    sensitivity=0.001,
    low_coverage=0.01,
)


ABLATION_VARIANTS = {

    "NF_BASE": AblationVariant(
        name="NF_BASE",
        uses_pso=False,
        dynamics=None,
        boundary=None,
        objective_weights=BASE_OBJECTIVE,
    ),

    "NFPSO": AblationVariant(
        name="NFPSO",
        uses_pso=True,
        dynamics="nonconstricted",
        boundary="feasible_rejection",
        objective_weights=BASE_OBJECTIVE,
    ),

    "P_NFPSO": AblationVariant(
        name="P_NFPSO",
        uses_pso=True,
        dynamics="nonconstricted",
        boundary="project",
        objective_weights=BASE_OBJECTIVE,
    ),

    "C_NFPSO": AblationVariant(
        name="C_NFPSO",
        uses_pso=True,
        dynamics="constricted",
        boundary="feasible_rejection",
        objective_weights=BASE_OBJECTIVE,
    ),

    "PC_NFPSO": AblationVariant(
        name="PC_NFPSO",
        uses_pso=True,
        dynamics="constricted",
        boundary="project",
        objective_weights=BASE_OBJECTIVE,
    ),

    "PC_NO_SENSITIVITY": AblationVariant(
        name="PC_NO_SENSITIVITY",
        uses_pso=True,
        dynamics="constricted",
        boundary="project",
        objective_weights=ObjectiveWeights(
            fit_rmse=0.5,
            validation_rmse=1.0,
            sensitivity=0.0,
            low_coverage=0.01,
        ),
    ),

    "PC_NO_COVERAGE": AblationVariant(
        name="PC_NO_COVERAGE",
        uses_pso=True,
        dynamics="constricted",
        boundary="project",
        objective_weights=ObjectiveWeights(
            fit_rmse=0.5,
            validation_rmse=1.0,
            sensitivity=0.001,
            low_coverage=0.0,
        ),
    ),

    "PC_VALIDATION_ONLY": AblationVariant(
        name="PC_VALIDATION_ONLY",
        uses_pso=True,
        dynamics="constricted",
        boundary="project",
        objective_weights=ObjectiveWeights(
            fit_rmse=0.0,
            validation_rmse=1.0,
            sensitivity=0.0,
            low_coverage=0.0,
        ),
    ),
}

def fit_ablation_variant(
    variant: AblationVariant,
    raw_pretest: np.ndarray,
    *,
    n_lags: int,
    validation_size: int,
    radius: float,
    alpha: float,
    optimizer_seed: int,
):
    """
    Unified V2 ablation dispatcher.

    NF_BASE:
        NF model without PSO.

    PSO variants:
        NFPSO / P_NFPSO / C_NFPSO / PC_NFPSO
        use the frozen PC-NFPSO engine with different dynamics/boundary/objective.
    """

    if not isinstance(variant, AblationVariant):
        raise TypeError("variant must be an AblationVariant.")

    if not variant.uses_pso:
        return fit_nf_base_v2(
            raw_pretest,
            n_lags=n_lags,
            validation_size=validation_size,
            radius=radius,
            alpha=alpha,
        )

    return fit_pc_nfpso_v2(
        raw_pretest,
        n_lags=n_lags,
        validation_size=validation_size,
        radius=radius,
        alpha=alpha,
        optimizer_seed=optimizer_seed,
        dynamics=variant.dynamics,
        boundary=variant.boundary,
        objective_weights=variant.objective_weights,
    )

__all__ = [
    "fit_nf_base_v2",
    "fit_ablation_variant",
    "AblationVariant",
    "ABLATION_VARIANTS",
]