# V2 Final External Benchmark Analysis Closure — 2026-08-14

## Status

**CLOSED.** The external Final benchmark and its pre-specified primary analysis
are complete. Ablation outcomes are not part of this closure and remain a
separate inferential family.

## Immutable execution identity

- Repository HEAD before this documentation commit: `ebfbb758a41c0146c9da618c4ce873976e4b5289`
- Canonical completed recovery workspace: `C:\Users\ADMIN\nfpsosc_v2_final_recovery_20260814`
- Final tasks: **8780 / 8780 successful**, 0 failed, 0 pending
- Synthetic tasks: **7140**
- Real tasks: **1640**
- Final result aggregate SHA-256:
  `393b7df1fca6d6dc2e57b60b06d6ccca7f15217e3a8c351b9427346829da299b`
- Frozen Final manifest SHA-256:
  `763da2f1e18a1f2572e97f7ece8b17438db0e20c3fe04ac0d05980208f28d5fd`
- Execution-freeze SHA-256:
  `750554cf28fdccffe8f3b7bae94b4cc6c761afbc4849c689b832f32e3ca768c2`

## Operational recovery incident

The original Final workspace remains immutable incident evidence at
`outputs\v2\final_external` with counts **8508 completed / 8504 successful / 4 failed /
272 pending**.

The four failed tasks were the first four PC-NFPSO optimizer replicates for
M4 Monthly series M16885. Each failed during **data loading** with host
`MemoryError`, before fitting or forecasting. The exact frozen tasks later
succeeded under one-worker recovery without changing scientific code, data,
series identity, optimizer seed, selected radius, selected ridge alpha, model
hyperparameters, or benchmark manifest. This strongly supports concurrent
memory pressure as the operational cause; it is not presented as a
mathematical proof of causality.

Incident manifest SHA-256:
`62bc0de2de2ee8c56451cf29f0afd87351246d8d5c02fc7f8af18a091c1bf8de`

## Analysis units

- Synthetic: **180 process × DGP-seed units**
- Real: **120 final real-series units**
- Seasonal-naive support: **40 synthetic**, **80 real**

Synthetic PC-NFPSO metrics use the median across five optimizer replicates
within each process × seed × noise condition, followed by the arithmetic mean
across noise levels 0.05, 0.10, and 0.20. Real PC-NFPSO metrics use the median
across five optimizer replicates within each series.

## Primary confirmatory MASE result

Paired difference:
`D = MASE_PC-NFPSO - MASE_baseline`; negative values favor PC-NFPSO.

Primary procedure:
- exact two-sided paired sign test;
- exact 95% nonparametric order-statistic CI for the paired median difference;
- Holm correction across the **fixed full family of 18 comparator-domain
  hypotheses**.

### Synthetic domain

PC-NFPSO showed Holm-confirmed superiority over:
`naive_1`, `seasonal_naive`, `drift`, `ets`, `theta`, `ridge_lag`, and
`xgboost`.

No directional superiority was established against:
`sarima` or `svr_rbf`.

### Real domain

PC-NFPSO showed Holm-confirmed superiority over:
`seasonal_naive`, `svr_rbf`, and `xgboost`.

The comparator showed Holm-confirmed superiority over PC-NFPSO for:
`naive_1`, `drift`, `sarima`, `ets`, `theta`, and `ridge_lag`.

A non-significant primary result is **not** interpreted as equivalence.

Primary-result-table SHA-256:
`ce650b1c95549ad714e6d09d59e7ecb385a007fb419a923c1fe26f224ad0c814`

## Secondary sensitivity analyses

### Wilcoxon

Wilcoxon signed-rank analysis was used only as a sensitivity analysis.
All 18 Holm-adjusted sensitivity comparisons were significant. This does not
replace or redefine the primary sign-test conclusions.

Wilcoxon table SHA-256:
`0b9ceb88bfbeb1fe5410ba7cf4165677560d5e9e88acd643d7f875ac84161660`

### Bootstrap arithmetic mean paired MASE difference

The frozen protocol specified 10,000 stratified resamples, seed 90210, and
95% confidence, but did not pre-specify the CI construction method or RNG
sequencing across comparisons. No single post-hoc interval was declared
canonical.

A 2×2 operational sensitivity matrix compared percentile vs basic intervals
under continuous-stream vs per-comparison RNG reset. All 18 comparisons agreed
across all four operationalizations on zero inclusion/exclusion and interval
direction. Sixteen excluded zero; `real/seasonal_naive` and `real/sarima`
crossed zero under all four.

Bootstrap operational-sensitivity table SHA-256:
`fcd10d1d255cd14f7ade16a071856211d2e7ae73b93b178247adbe5697b8ce8f`

## Secondary metrics

Secondary descriptive metrics were MAE, RMSE, sMAPE, and RMSSE.

Raw-scale MAE/RMSE magnitudes are not used as pooled cross-scale headline
statistics. Finite extreme forecasts remain in the analysis without clipping,
Winsorization, removal, or silent substitution.

The M3 Monthly M90 SARIMA forecast is retained as a catastrophic-but-finite
instability and materially distorts arithmetic mean summaries. Robust primary
median/sign-test inference therefore remains the confirmatory basis.

Secondary-table hashes:
- scale-free: `e797629cf10375a457e4639bdbb4e2c3ba5805b196e932c4af6fc00291f25457`
- raw-scale direction counts: `16d20926ff394f70fb75815961322913a517c895102af47c83a6931bc053a1bc`
- raw-scale stratified: `e138ad1a89d4c70108f8894c9a08adf68230316040dd541797056ef36121b894`
- top-20 extremes: `2a1ff2d44c992ea82ec336ed2cf1e3d43335c4d23631f6959487fde32238c6c5`

## Scope boundary

This closure freezes the interpretation of the external Final benchmark.
The pre-specified internal **ablation study remains separate**, uses its own
seven-contrast Holm family, and must be implemented/frozen before any ablation
outcomes are generated.
