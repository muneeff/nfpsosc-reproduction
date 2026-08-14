# Protocol Amendment V2-A009

## Final stochastic aggregation and secondary-metric operationalization

Date: 2026-08-14

Status: LOCKED PRE-FINAL PERFORMANCE

### Trigger

This clarification was made after Development selection had been completed and frozen, and before any Final synthetic or Final real forecast performance outcome had been generated.

The frozen protocol already specified five Final optimizer seeds for PC-NFPSO and defined the real confirmatory analysis unit as one series. However, it did not explicitly state how the five stochastic PC-NFPSO replicates should be reduced to one real-series metric value. The protocol also listed sMAPE and RMSSE as secondary metrics without fully specifying their numerical formulas.

### PC-NFPSO optimizer-replicate aggregation

For the synthetic Final benchmark, use optimizer seeds 41001, 41002, 41003, 41004, and 41005. Compute each metric separately for each optimizer replicate. Within each process × DGP-seed × noise condition, take the median of the five metric values. Do not aggregate predictions. Then take the arithmetic mean across noise levels 0.05, 0.10, and 0.20 to obtain the process × DGP-seed analysis-unit metric.

For the real Final benchmark, use the same five optimizer seeds for every frozen final series. Compute each metric separately for each optimizer replicate and take the median of the five metric values to obtain one metric value for that real-series analysis unit. Do not aggregate predictions.

### Baseline aggregation

Baselines have no PC-NFPSO optimizer-replicate dimension.

For synthetic data, run each baseline once for each process × DGP-seed × noise condition, compute the metric for that condition, and then take the arithmetic mean across the three confirmatory noise levels to obtain one process × DGP-seed analysis-unit metric.

For real data, run each baseline once per frozen final real series.

Seasonal-naive comparison rows exist only where the frozen seasonal period is greater than one.

### Secondary metric formulas

MAE:

`mean(abs(y-yhat))`

RMSE:

`sqrt(mean((y-yhat)^2))`

sMAPE:

`200 * mean(abs(y-yhat) / (abs(y)+abs(yhat)))`

sMAPE is reported in percent. A forecast-origin term with both actual and predicted values equal to zero contributes zero.

RMSSE:

`sqrt(mean_test((y-yhat)^2) / mean_train((y[t]-y[t-m])^2))`

The RMSSE scaling denominator uses only the pre-test training segment and the frozen seasonal period. The numerical minimum squared scale is `1e-24`.

The existing MASE definition remains unchanged, including its training-only denominator and minimum scale of `1e-12`.

### Failure and estimability rule

All five required PC-NFPSO optimizer replicates must succeed and produce finite predictions and finite metrics before a PC-NFPSO condition or real-series metric is estimable. A median must not be formed from a surviving subset of optimizer replicates.

For a synthetic process × DGP-seed analysis unit, all three confirmatory noise-condition aggregates must be estimable before noise aggregation.

A baseline synthetic analysis unit likewise requires successful values for all three confirmatory noise levels.

Pairwise accuracy analysis uses only frozen analysis units for which both PC-NFPSO and the comparator are estimable. Model failures and failure rates are reported separately. No failed forecast is imputed, substituted, or replaced by another model.

### Scientific scope

V2-A009 does not change the selected radius (`1.0`), selected ridge alpha (`0.05`), Final DGP seeds, optimizer seeds, real-series identities, baseline configurations, or inferential families.

It is a performance-blind operational clarification made before any Final performance outcome is generated.
