"""
Statistical comparison utilities for the SS-NFPSO Q1 experiment pipeline.

The functions in this module are designed for leakage-aware forecasting
experiments where every model is evaluated on the same datasets, seeds, or
controlled experimental conditions.

Key principles
--------------
- Use paired comparisons whenever possible.
- Treat lower error metrics as better by default.
- Keep SciPy optional: if SciPy is available, use Friedman/Wilcoxon tests;
  otherwise provide conservative fallback outputs instead of crashing.
- Do not create publication claims automatically; these functions only compute
  evidence summaries that must be interpreted in the paper.

Typical input
-------------
A metrics table with at least:

    dataset, model, rmse

or, for synthetic experiments:

    dataset, generator, noise_level, seed, model, rmse

Examples
--------
>>> import pandas as pd
>>> from nfpsosc.statistics import summarize_by_model, compare_against_reference
>>> df = pd.DataFrame({
...     "dataset": ["a", "a", "b", "b"],
...     "model": ["m1", "m2", "m1", "m2"],
...     "rmse": [1.0, 1.4, 2.0, 1.8],
... })
>>> summarize_by_model(df, metric="rmse")[["model", "mean", "rank_mean"]]
  model  mean  rank_mean
0    m1   1.5        1.5
1    m2   1.6        1.5
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, erf, sqrt
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ERROR_METRICS_LOWER_IS_BETTER = {
    "mse",
    "rmse",
    "mae",
    "mape_fraction",
    "mape_percent",
    "smape_fraction",
    "smape_percent",
    "mase",
    "error_std_sample",
    "runtime_seconds",
}

SCORE_METRICS_HIGHER_IS_BETTER = {
    "r2",
}


@dataclass(frozen=True)
class TestResult:
    """Small serializable container for statistical test outputs."""

    test_name: str
    statistic: Optional[float]
    p_value: Optional[float]
    n_pairs: Optional[int]
    method: str
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation."""

        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n_pairs": self.n_pairs,
            "method": self.method,
            "note": self.note,
        }


def lower_is_better_metric(metric: str) -> bool:
    """Return whether a metric should be minimized."""

    metric_lower = metric.lower()
    if metric_lower in SCORE_METRICS_HIGHER_IS_BETTER:
        return False
    return True


def normal_two_sided_p_value(z: float) -> float:
    """Two-sided normal p-value using only the Python standard math functions."""

    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    phi = 0.5 * (1.0 + erf(abs(float(z)) / sqrt(2.0)))
    return float(2.0 * (1.0 - phi))


def make_condition_column(
    df: pd.DataFrame,
    condition_cols: Optional[Sequence[str]] = None,
    fallback_col: str = "dataset",
) -> pd.Series:
    """
    Build a paired-condition key.

    If condition_cols is omitted, this function uses 'dataset' when present.
    For synthetic experiments, a stricter condition can be created from columns
    such as ['generator', 'length', 'noise_level', 'seed'].
    """

    if condition_cols is None:
        if fallback_col in df.columns:
            return df[fallback_col].astype(str)
        raise ValueError(f"Missing fallback condition column: {fallback_col}")

    missing = [c for c in condition_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing condition columns: {missing}")

    return df[list(condition_cols)].astype(str).agg(" | ".join, axis=1)


def pivot_metric(
    df: pd.DataFrame,
    metric: str,
    model_col: str = "model",
    condition_cols: Optional[Sequence[str]] = None,
    condition_col_name: str = "_condition",
    aggregate: str = "mean",
) -> pd.DataFrame:
    """
    Convert a long metrics table into condition x model format.

    Duplicate condition-model pairs are aggregated by mean by default.
    """

    if metric not in df.columns:
        raise ValueError(f"Metric column not found: {metric}")
    if model_col not in df.columns:
        raise ValueError(f"Model column not found: {model_col}")

    working = df.copy()
    working[condition_col_name] = make_condition_column(working, condition_cols)
    working[metric] = pd.to_numeric(working[metric], errors="coerce")
    working = working.dropna(subset=[metric, model_col, condition_col_name])

    if working.empty:
        raise ValueError("No valid rows after cleaning metric table.")

    if aggregate == "mean":
        grouped = (
            working.groupby([condition_col_name, model_col], as_index=False)[metric]
            .mean()
        )
    elif aggregate == "median":
        grouped = (
            working.groupby([condition_col_name, model_col], as_index=False)[metric]
            .median()
        )
    else:
        raise ValueError("aggregate must be 'mean' or 'median'.")

    wide = grouped.pivot(index=condition_col_name, columns=model_col, values=metric)
    wide = wide.sort_index(axis=0).sort_index(axis=1)
    return wide


def complete_case_wide(wide: pd.DataFrame, min_models: int = 2) -> pd.DataFrame:
    """
    Drop conditions with missing model values.

    Statistical tests such as Friedman require complete paired rows.
    """

    if wide.shape[1] < min_models:
        raise ValueError(f"At least {min_models} models are required.")
    complete = wide.dropna(axis=0, how="any")
    if complete.shape[0] == 0:
        raise ValueError("No complete paired conditions are available.")
    return complete


def rank_models(
    wide: pd.DataFrame,
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """Rank models within each condition."""

    ascending = bool(lower_is_better)
    return wide.rank(axis=1, method="average", ascending=ascending)


def summarize_by_model(
    df: pd.DataFrame,
    metric: str,
    model_col: str = "model",
    condition_cols: Optional[Sequence[str]] = None,
    lower_is_better: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Summarize model performance and average ranks.

    Returns one row per model with mean, std, median, count, and mean rank.
    """

    if lower_is_better is None:
        lower_is_better = lower_is_better_metric(metric)

    wide = pivot_metric(df, metric=metric, model_col=model_col, condition_cols=condition_cols)
    ranks = rank_models(wide, lower_is_better=lower_is_better)

    summary = []
    for model in wide.columns:
        values = pd.to_numeric(wide[model], errors="coerce").dropna()
        rank_values = pd.to_numeric(ranks[model], errors="coerce").dropna()

        summary.append(
            {
                "model": model,
                "metric": metric,
                "mean": float(values.mean()) if len(values) else np.nan,
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "median": float(values.median()) if len(values) else np.nan,
                "min": float(values.min()) if len(values) else np.nan,
                "max": float(values.max()) if len(values) else np.nan,
                "n_conditions": int(len(values)),
                "rank_mean": float(rank_values.mean()) if len(rank_values) else np.nan,
                "rank_median": float(rank_values.median()) if len(rank_values) else np.nan,
                "lower_is_better": bool(lower_is_better),
            }
        )

    out = pd.DataFrame(summary)
    return out.sort_values(["rank_mean", "mean"], ascending=[True, bool(lower_is_better)]).reset_index(drop=True)


def friedman_test(
    df: pd.DataFrame,
    metric: str,
    model_col: str = "model",
    condition_cols: Optional[Sequence[str]] = None,
    lower_is_better: Optional[bool] = None,
) -> TestResult:
    """
    Run a Friedman test across models.

    Uses scipy.stats.friedmanchisquare when SciPy is installed.
    """

    if lower_is_better is None:
        lower_is_better = lower_is_better_metric(metric)

    wide = pivot_metric(df, metric=metric, model_col=model_col, condition_cols=condition_cols)
    complete = complete_case_wide(wide, min_models=3)

    if complete.shape[1] < 3:
        raise ValueError("Friedman test requires at least three models.")

    try:
        from scipy.stats import friedmanchisquare  # type: ignore

        values = [complete[col].to_numpy(dtype=float) for col in complete.columns]
        stat, p_value = friedmanchisquare(*values)
        return TestResult(
            test_name="friedman",
            statistic=float(stat),
            p_value=float(p_value),
            n_pairs=int(complete.shape[0]),
            method="scipy.stats.friedmanchisquare",
            note=f"complete_conditions={complete.shape[0]}, models={complete.shape[1]}",
        )
    except Exception as exc:
        # Fallback: compute Friedman chi-square statistic, without exact p-value.
        ranks = rank_models(complete, lower_is_better=lower_is_better)
        n = complete.shape[0]
        k = complete.shape[1]
        rank_sums = ranks.sum(axis=0).to_numpy(dtype=float)
        stat = (12.0 / (n * k * (k + 1.0))) * np.sum(rank_sums ** 2) - 3.0 * n * (k + 1.0)
        return TestResult(
            test_name="friedman",
            statistic=float(stat),
            p_value=None,
            n_pairs=int(n),
            method="fallback_friedman_statistic_without_p_value",
            note=f"SciPy unavailable or failed: {type(exc).__name__}: {exc}",
        )


def paired_differences(
    df: pd.DataFrame,
    reference_model: str,
    challenger_model: str,
    metric: str,
    model_col: str = "model",
    condition_cols: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    Return paired challenger - reference differences.

    For error metrics, negative values mean the challenger is better.
    """

    wide = pivot_metric(df, metric=metric, model_col=model_col, condition_cols=condition_cols)
    required = [reference_model, challenger_model]
    missing = [m for m in required if m not in wide.columns]
    if missing:
        raise ValueError(f"Missing models in paired table: {missing}")

    paired = wide[required].dropna(axis=0, how="any")
    if paired.empty:
        raise ValueError("No paired rows for the requested model comparison.")

    diff = paired[challenger_model].to_numpy(dtype=float) - paired[reference_model].to_numpy(dtype=float)
    return diff


def exact_sign_test_p_value(differences: np.ndarray) -> Tuple[float, int]:
    """
    Conservative exact two-sided sign-test p-value.

    Zero differences are ignored.
    """

    nonzero = differences[np.abs(differences) > 0]
    n = int(len(nonzero))
    if n == 0:
        return 1.0, 0

    positives = int(np.sum(nonzero > 0))
    negatives = int(np.sum(nonzero < 0))
    k = min(positives, negatives)

    # Two-sided binomial test with p=0.5.
    prob = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    p_value = min(1.0, 2.0 * prob)
    return float(p_value), n


def wilcoxon_signed_rank_test(
    differences: np.ndarray,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Run paired Wilcoxon signed-rank test.

    Falls back to an exact sign test if SciPy is not available.
    """

    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]

    if len(differences) == 0:
        raise ValueError("No finite paired differences.")

    try:
        from scipy.stats import wilcoxon  # type: ignore

        stat, p_value = wilcoxon(differences, alternative=alternative, zero_method="wilcox")
        return TestResult(
            test_name="wilcoxon_signed_rank",
            statistic=float(stat),
            p_value=float(p_value),
            n_pairs=int(len(differences)),
            method="scipy.stats.wilcoxon",
            note=f"alternative={alternative}",
        )
    except Exception as exc:
        p_value, n_nonzero = exact_sign_test_p_value(differences)
        return TestResult(
            test_name="sign_test_fallback",
            statistic=float(np.sum(differences > 0)),
            p_value=float(p_value),
            n_pairs=int(n_nonzero),
            method="exact_two_sided_sign_test",
            note=f"Wilcoxon unavailable or failed: {type(exc).__name__}: {exc}",
        )


def bootstrap_mean_difference(
    differences: np.ndarray,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 123,
) -> Dict[str, Any]:
    """
    Bootstrap confidence interval for the paired mean difference.

    For error metrics, challenger - reference < 0 favors challenger.
    """

    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0:
        raise ValueError("No finite paired differences for bootstrap.")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(int(n_resamples), len(differences)))
    samples = differences[indices].mean(axis=1)

    alpha = 1.0 - confidence
    low = float(np.quantile(samples, alpha / 2.0))
    high = float(np.quantile(samples, 1.0 - alpha / 2.0))

    return {
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "ci_low": low,
        "ci_high": high,
        "confidence": float(confidence),
        "n_pairs": int(len(differences)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def holm_correction(p_values: Sequence[float], alpha: float = 0.05) -> pd.DataFrame:
    """
    Holm-Bonferroni correction for multiple comparisons.

    Returns adjusted p-values and reject decisions in the original order.
    """

    p_values_arr = np.asarray(p_values, dtype=float)
    m = len(p_values_arr)
    order = np.argsort(p_values_arr)

    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = (m - rank) * p_values_arr[idx]
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = min(1.0, running_max)

    adjusted_original = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        adjusted_original[idx] = adjusted_sorted[rank]

    return pd.DataFrame(
        {
            "comparison_index": list(range(m)),
            "p_value": p_values_arr,
            "p_adjusted_holm": adjusted_original,
            "reject_alpha": adjusted_original < float(alpha),
            "alpha": float(alpha),
        }
    )


def compare_against_reference(
    df: pd.DataFrame,
    reference_model: str,
    metric: str,
    model_col: str = "model",
    condition_cols: Optional[Sequence[str]] = None,
    lower_is_better: Optional[bool] = None,
    alpha: float = 0.05,
    bootstrap_resamples: int = 10000,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Compare every model against a reference model with paired tests.

    For lower-is-better metrics:
    - mean_difference = challenger - reference
    - negative mean_difference means challenger is better than reference
    - positive mean_difference means reference is better than challenger
    """

    if lower_is_better is None:
        lower_is_better = lower_is_better_metric(metric)

    wide = pivot_metric(df, metric=metric, model_col=model_col, condition_cols=condition_cols)
    if reference_model not in wide.columns:
        raise ValueError(f"Reference model not found: {reference_model}")

    rows: List[Dict[str, Any]] = []
    p_values: List[float] = []

    for challenger in wide.columns:
        if challenger == reference_model:
            continue

        diff = paired_differences(
            df,
            reference_model=reference_model,
            challenger_model=str(challenger),
            metric=metric,
            model_col=model_col,
            condition_cols=condition_cols,
        )

        test = wilcoxon_signed_rank_test(diff)
        boot = bootstrap_mean_difference(
            diff,
            n_resamples=bootstrap_resamples,
            confidence=1.0 - alpha,
            seed=seed,
        )

        mean_diff = boot["mean_difference"]
        if lower_is_better:
            direction = (
                "challenger_better"
                if mean_diff < 0
                else "reference_better"
                if mean_diff > 0
                else "tie"
            )
        else:
            direction = (
                "challenger_better"
                if mean_diff > 0
                else "reference_better"
                if mean_diff < 0
                else "tie"
            )

        row = {
            "reference_model": reference_model,
            "challenger_model": challenger,
            "metric": metric,
            "lower_is_better": bool(lower_is_better),
            "direction_by_mean": direction,
            **test.as_dict(),
            **boot,
        }
        rows.append(row)
        p_values.append(float(test.p_value) if test.p_value is not None else np.nan)

    out = pd.DataFrame(rows)
    if not out.empty and np.all(np.isfinite(p_values)):
        correction = holm_correction(p_values, alpha=alpha)
        out["p_adjusted_holm"] = correction["p_adjusted_holm"].values
        out["reject_alpha_holm"] = correction["reject_alpha"].values
    else:
        out["p_adjusted_holm"] = np.nan
        out["reject_alpha_holm"] = False

    return out


def critical_difference_nemenyi(
    n_models: int,
    n_conditions: int,
    q_alpha: Optional[float] = None,
) -> Optional[float]:
    """
    Approximate Nemenyi critical difference for average ranks.

    This function requires q_alpha from a Studentized range table. It does not
    guess the value because the proper value depends on alpha and model count.
    Pass q_alpha explicitly when preparing a manuscript figure.

    CD = q_alpha * sqrt(k(k+1)/(6N))
    """

    if q_alpha is None:
        return None
    if n_models <= 1 or n_conditions <= 0:
        raise ValueError("n_models must be > 1 and n_conditions must be > 0.")
    return float(q_alpha * sqrt((n_models * (n_models + 1.0)) / (6.0 * n_conditions)))


def export_statistical_report(
    df: pd.DataFrame,
    output_dir: str | Path,
    metric: str,
    reference_model: Optional[str] = None,
    model_col: str = "model",
    condition_cols: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    bootstrap_resamples: int = 10000,
    seed: int = 123,
) -> Dict[str, str]:
    """
    Export summary, Friedman, and reference-comparison CSV files.

    Returns a dictionary of output file paths.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}

    summary = summarize_by_model(
        df,
        metric=metric,
        model_col=model_col,
        condition_cols=condition_cols,
    )
    summary_path = output_path / f"{metric}_model_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    paths["summary"] = str(summary_path)

    try:
        friedman = friedman_test(
            df,
            metric=metric,
            model_col=model_col,
            condition_cols=condition_cols,
        )
        friedman_df = pd.DataFrame([friedman.as_dict()])
    except Exception as exc:
        friedman_df = pd.DataFrame(
            [
                {
                    "test_name": "friedman",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "n_pairs": np.nan,
                    "method": "not_run",
                    "note": f"{type(exc).__name__}: {exc}",
                }
            ]
        )
    friedman_path = output_path / f"{metric}_friedman.csv"
    friedman_df.to_csv(friedman_path, index=False, encoding="utf-8-sig")
    paths["friedman"] = str(friedman_path)

    if reference_model is not None:
        comparisons = compare_against_reference(
            df,
            reference_model=reference_model,
            metric=metric,
            model_col=model_col,
            condition_cols=condition_cols,
            alpha=alpha,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
        comparisons_path = output_path / f"{metric}_reference_comparisons.csv"
        comparisons.to_csv(comparisons_path, index=False, encoding="utf-8-sig")
        paths["reference_comparisons"] = str(comparisons_path)

    return paths


__all__ = [
    "TestResult",
    "bootstrap_mean_difference",
    "compare_against_reference",
    "complete_case_wide",
    "critical_difference_nemenyi",
    "export_statistical_report",
    "friedman_test",
    "holm_correction",
    "lower_is_better_metric",
    "paired_differences",
    "pivot_metric",
    "rank_models",
    "summarize_by_model",
    "wilcoxon_signed_rank_test",
]
