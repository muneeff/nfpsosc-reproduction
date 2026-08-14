# V2 Primary Ablation Inference

**Archive date:** 2026-08-15

**Status:** `PRIMARY_INFERENCE_ARCHIVED`

## Frozen family

- Domain: synthetic
- Analysis units: 90 process × DGP-seed units
- Reference: PC_NFPSO
- Metric: MASE
- Effect: D = MASE_PC_NFPSO - MASE_variant
- Primary test: exact two-sided sign test
- Median interval: exact 95% order-statistic interval
- Holm family: exactly 7 contrasts
- Familywise alpha: 0.05

## Primary results

| Comparator | Wins PC | Ties | Losses PC | Median D | 95% median CI | Raw p | Holm p | Directional conclusion |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| NF_BASE | 44 | 15 | 31 | 0 | [-0.0437574, 0] | 0.165428 | 0.992568 | No supported superiority |
| NFPSO | 37 | 15 | 38 | 0 | [-0.00238196, 0.000357025] | 1 | 1 | No supported superiority |
| P_NFPSO | 40 | 15 | 35 | 0 | [-0.00316671, 0] | 0.644464 | 1 | No supported superiority |
| C_NFPSO | 33 | 16 | 41 | 0 | [0, 0.00161898] | 0.415985 | 1 | No supported superiority |
| PC_NO_SENSITIVITY | 41 | 15 | 34 | 0 | [-0.00326609, 0] | 0.488683 | 1 | No supported superiority |
| PC_NO_COVERAGE | 31 | 34 | 25 | 0 | [0, 0] | 0.504404 | 1 | No supported superiority |
| PC_VALIDATION_ONLY | 48 | 15 | 27 | -0.00615196 | [-0.0146819, 0] | 0.0202984 | 0.142089 | No supported superiority |

Primary inference table SHA-256:
`c7e2f5f8ee147d36e6c1723697df707ff1b9a7f97ccf372732f6fe15112d4516`

## Family-level interpretation

None of the seven pre-specified contrasts is significant after the frozen Holm
correction. Therefore the primary ablation family provides no statistically
supported directional superiority claim for PC_NFPSO over any tested variant,
and no supported superiority claim in the opposite direction.

PC_VALIDATION_ONLY shows the strongest unadjusted tendency favoring PC_NFPSO
(raw exact sign-test p = 0.0202984; median D = -0.00615196), but its Holm-
adjusted p = 0.142089. It must therefore be treated as suggestive rather than
confirmatory.

Failure to reject these contrasts is not evidence of equivalence, no effect,
or dispensability of the corresponding components.
