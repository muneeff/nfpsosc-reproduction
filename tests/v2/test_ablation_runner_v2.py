import numpy as np

from nfpsosc.v2.ablation_runner import (
    ABLATION_VARIANTS,
    AblationVariant,
    fit_ablation_variant,
)


def test_ablation_variant_registry():
    expected = {
        "NF_BASE",
        "NFPSO",
        "P_NFPSO",
        "C_NFPSO",
        "PC_NFPSO",
        "PC_NO_SENSITIVITY",
        "PC_NO_COVERAGE",
        "PC_VALIDATION_ONLY",
    }

    assert set(ABLATION_VARIANTS.keys()) == expected


def test_nf_base_configuration():
    variant = ABLATION_VARIANTS["NF_BASE"]

    assert variant.uses_pso is False
    assert variant.dynamics is None
    assert variant.boundary is None


def test_pc_nfpso_configuration():
    variant = ABLATION_VARIANTS["PC_NFPSO"]

    assert variant.uses_pso is True
    assert variant.dynamics == "constricted"
    assert variant.boundary == "project"


def test_ablation_dispatch_nf_base():
    raw = np.sin(np.linspace(0, 10, 120))

    result = fit_ablation_variant(
        ABLATION_VARIANTS["NF_BASE"],
        raw,
        n_lags=5,
        validation_size=20,
        radius=0.5,
        alpha=0.01,
        optimizer_seed=1,
    )

    assert "model" in result
    assert "training" in result


def test_ablation_dispatch_pc_nfpso():
    raw = np.sin(np.linspace(0, 10, 120))

    result = fit_ablation_variant(
        ABLATION_VARIANTS["PC_NFPSO"],
        raw,
        n_lags=5,
        validation_size=20,
        radius=0.5,
        alpha=0.01,
        optimizer_seed=1,
    )

    assert hasattr(result, "final_model")
    assert result.dynamics == "constricted"
    assert result.boundary == "project"