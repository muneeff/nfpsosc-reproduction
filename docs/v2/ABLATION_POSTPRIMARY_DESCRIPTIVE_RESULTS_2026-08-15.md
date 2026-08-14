# V2 Post-Primary Ablation Descriptive Results

**Archive date:** 2026-08-15

**Status:** `POSTPRIMARY_DESCRIPTIVE_RESULTS_ARCHIVED`

## Confirmatory boundary

The confirmatory ablation conclusion remains unchanged: **0 of 7**
pre-specified contrasts were significant after Holm correction.

Everything below is descriptive only.

## Synthetic overall medians

| Variant | Median unit MASE |
|---|---:|
| PC_NFPSO | 0.8111914145 |
| NF_BASE | 0.8083598760 |
| NFPSO | 0.8172682656 |
| P_NFPSO | 0.8219174676 |
| C_NFPSO | 0.8087931422 |
| PC_NO_SENSITIVITY | 0.8266691245 |
| PC_NO_COVERAGE | 0.8190250808 |
| PC_VALIDATION_ONLY | 0.8250212933 |

These marginal medians do not define a paired inferential comparison and must
not be used to replace the frozen paired primary analysis.

## Descriptive heterogeneity

Generator-level patterns are heterogeneous. PC_NFPSO is descriptively strong
against NF_BASE on Hénon, logistic-map, Lorenz63, and nonlinear-sine
generators, while NF_BASE is descriptively stronger on noisy-seasonal and
regime-switching generators. Short-memory and spiky-outbreak contain many
exact ties.

Against PC_VALIDATION_ONLY, several nonlinear generators show negative median
D values favoring PC_NFPSO, but this remains descriptive; the primary
Holm-adjusted contrast was not significant.

No generator-level p-value or confidence interval was computed.

## Real descriptive anchor: PC_NFPSO vs NF_BASE

- Series: 120
- PC_NFPSO better: 39
- Exact ties: 9
- NF_BASE better: 72
- Median D: 0.0528447168
- Median MASE PC_NFPSO: 0.9491817702
- Median MASE NF_BASE: 0.9374341217

Because `D = MASE_PC_NFPSO - MASE_NF_BASE`, the positive median D and
72-versus-39 count are descriptively in the direction of NF_BASE on the real
anchor. No p-value or confidence interval was computed, and this comparison is
not an eighth formal test.

## Frozen output hashes

- Synthetic overall descriptive SHA-256:
  `a35e27f2345d2dba11ac768cac9f0efdb7c2fea079bdb1f1a75fa8909250a22f`
- Synthetic generator descriptive SHA-256:
  `981436def20be8fa458cd9a224636577ad34c434c860fb84e3b04f4120bc7aa4`
- Real anchor summary SHA-256:
  `317b0330ae877f70d11903a599b88e29383aead74f8c294ededfe7c6821b8c86`
- Real series descriptive table SHA-256:
  `d9af70ab60bf211cfc8ad567fe1936e1b545316305716c81701d46892a1f4bb5`

No new hypothesis test, Wilcoxon test, bootstrap, confidence interval, or
secondary-metric inference is introduced by this archive.
