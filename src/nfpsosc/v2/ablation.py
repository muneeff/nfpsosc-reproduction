from __future__ import annotations

"""Frozen V2 internal-ablation variant layer.

Maps the pre-specified ablation variants onto the already frozen V2
training/optimizer implementation. No new optimizer, feature representation,
hyperparameter search, or test-time update policy is introduced.

The PC_NFPSO reference can be fitted through this API for regression testing,
but production ablation execution reuses the cryptographically closed Final
reference rows per V2-A010.
"""

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from .initialization import (
    TSKInitialization,
    initialize_tsk_from_subtractive_clustering,
)
from .model import TSKModel
from .objective import ObjectiveWeights
from .pc_nfpso import (
    PCNFPSOFit,
    PCNFPSOTrainingData,
    fit_pc_nfpso_v2,
    forecast_pc_nfpso_v2,
    prepare_pc_nfpso_training_data,
)

AblationVariant = Literal[
    "NF_BASE",
    "NFPSO",
    "P_NFPSO",
    "C_NFPSO",
    "PC_NFPSO",
    "PC_NO_SENSITIVITY",
    "PC_NO_COVERAGE",
    "PC_VALIDATION_ONLY",
]

VARIANT_ORDER: tuple[str, ...] = (
    "NF_BASE",
    "NFPSO",
    "P_NFPSO",
    "C_NFPSO",
    "PC_NFPSO",
    "PC_NO_SENSITIVITY",
    "PC_NO_COVERAGE",
    "PC_VALIDATION_ONLY",
)

@dataclass(frozen=True)
class AblationVariantSpec:
    name: str
    pso: bool
    dynamics: Literal["nonconstricted", "constricted"] | None
    boundary: Literal["project", "feasible_rejection"] | None
    objective_weights: ObjectiveWeights | None

FULL_OBJECTIVE = ObjectiveWeights()
NO_SENSITIVITY_OBJECTIVE = ObjectiveWeights(sensitivity=0.0)
NO_COVERAGE_OBJECTIVE = ObjectiveWeights(low_coverage=0.0)
VALIDATION_ONLY_OBJECTIVE = ObjectiveWeights(
    fit_rmse=0.0,
    validation_rmse=1.0,
    sensitivity=0.0,
    low_coverage=0.0,
)

VARIANT_SPECS: dict[str, AblationVariantSpec] = {
    "NF_BASE": AblationVariantSpec("NF_BASE", False, None, None, None),
    "NFPSO": AblationVariantSpec(
        "NFPSO", True, "nonconstricted", "feasible_rejection", FULL_OBJECTIVE
    ),
    "P_NFPSO": AblationVariantSpec(
        "P_NFPSO", True, "nonconstricted", "project", FULL_OBJECTIVE
    ),
    "C_NFPSO": AblationVariantSpec(
        "C_NFPSO", True, "constricted", "feasible_rejection", FULL_OBJECTIVE
    ),
    "PC_NFPSO": AblationVariantSpec(
        "PC_NFPSO", True, "constricted", "project", FULL_OBJECTIVE
    ),
    "PC_NO_SENSITIVITY": AblationVariantSpec(
        "PC_NO_SENSITIVITY", True, "constricted", "project",
        NO_SENSITIVITY_OBJECTIVE
    ),
    "PC_NO_COVERAGE": AblationVariantSpec(
        "PC_NO_COVERAGE", True, "constricted", "project",
        NO_COVERAGE_OBJECTIVE
    ),
    "PC_VALIDATION_ONLY": AblationVariantSpec(
        "PC_VALIDATION_ONLY", True, "constricted", "project",
        VALIDATION_ONLY_OBJECTIVE
    ),
}

@dataclass(frozen=True)
class NFBaseFit:
    """Deterministic clustering-only neuro-fuzzy reference fit."""
    training: PCNFPSOTrainingData
    initialization: TSKInitialization
    final_model: TSKModel
    radius: float
    alpha: float
    optimizer_seed: None = None
    dynamics: None = None
    boundary: None = None
    variant: str = "NF_BASE"

AblationFit = PCNFPSOFit | NFBaseFit

def variant_spec(variant: str) -> AblationVariantSpec:
    name = str(variant)
    try:
        return VARIANT_SPECS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown frozen V2 ablation variant: {name!r}."
        ) from exc

def fit_nf_base_v2(
    raw_pretest: np.ndarray,
    *,
    n_lags: int,
    validation_size: int,
    radius: float,
    alpha: float,
) -> NFBaseFit:
    """Fit deterministic NF_BASE without invoking PSO."""
    if not np.isfinite(radius) or float(radius) <= 0.0:
        raise ValueError("radius must be a finite positive scalar.")
    if not np.isfinite(alpha) or float(alpha) <= 0.0:
        raise ValueError("alpha must be a finite positive scalar.")

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

    final_model = initialization.model.copy()
    final_model.fit_consequents(
        training.X_pretest,
        training.y_pretest,
        alpha=float(alpha),
    )

    return NFBaseFit(
        training=training,
        initialization=initialization,
        final_model=final_model,
        radius=float(radius),
        alpha=float(alpha),
    )

def fit_ablation_variant_v2(
    variant: str,
    raw_pretest: np.ndarray,
    *,
    n_lags: int,
    validation_size: int,
    radius: float,
    alpha: float,
    optimizer_seed: int | None,
) -> AblationFit:
    """Dispatch exactly one frozen V2 ablation variant."""
    spec = variant_spec(variant)

    if not spec.pso:
        if optimizer_seed is not None:
            raise ValueError(
                "NF_BASE is deterministic and cannot carry an optimizer seed."
            )
        return fit_nf_base_v2(
            raw_pretest,
            n_lags=n_lags,
            validation_size=validation_size,
            radius=radius,
            alpha=alpha,
        )

    if optimizer_seed is None:
        raise ValueError(f"{spec.name} requires a frozen optimizer seed.")

    assert spec.dynamics is not None
    assert spec.boundary is not None
    assert spec.objective_weights is not None

    return fit_pc_nfpso_v2(
        raw_pretest,
        n_lags=n_lags,
        validation_size=validation_size,
        radius=radius,
        alpha=alpha,
        optimizer_seed=int(optimizer_seed),
        dynamics=spec.dynamics,
        boundary=spec.boundary,
        objective_weights=spec.objective_weights,
    )

def forecast_ablation_v2(
    fitted: AblationFit,
    test_actuals: np.ndarray,
) -> np.ndarray:
    """Reuse the exact frozen observed-history one-step forecast implementation."""
    return forecast_pc_nfpso_v2(
        cast(PCNFPSOFit, fitted),
        test_actuals,
    )

__all__ = [
    "AblationVariant",
    "AblationVariantSpec",
    "NFBaseFit",
    "AblationFit",
    "VARIANT_ORDER",
    "VARIANT_SPECS",
    "FULL_OBJECTIVE",
    "NO_SENSITIVITY_OBJECTIVE",
    "NO_COVERAGE_OBJECTIVE",
    "VALIDATION_ONLY_OBJECTIVE",
    "variant_spec",
    "fit_nf_base_v2",
    "fit_ablation_variant_v2",
    "forecast_ablation_v2",
]
