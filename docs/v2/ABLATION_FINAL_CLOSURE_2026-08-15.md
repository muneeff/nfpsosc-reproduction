# V2 Final Ablation Closure

**Closure date:** 2026-08-15

**Status:** `FINAL_ABLATION_CLOSED`

## Provenance

- New ablation outcomes: **5,250 / 5,250 successful**
- Failed: **0**
- Pending: **0**
- Reused frozen PC_NFPSO reference rows: **1,410**
- Result-corpus SHA-256: `3628d71f833bd50ae9c5ff8799568e159bb89afa525e6d3bc5bafc77bc44a45b`
- External Final aggregate SHA-256: `393b7df1fca6d6dc2e57b60b06d6ccca7f15217e3a8c351b9427346829da299b`

## Frozen primary analysis

- Synthetic analysis units: **90**
- Reference: `PC_NFPSO`
- Formal contrasts: **7**
- Primary metric: `MASE`
- Primary inference table SHA-256: `c7e2f5f8ee147d36e6c1723697df707ff1b9a7f97ccf372732f6fe15112d4516`

## Confirmatory conclusion

**0 of 7 pre-specified ablation contrasts were Holm-significant.**

No familywise-supported directional superiority claim is justified for
PC_NFPSO over any tested variant, or for any tested variant over PC_NFPSO.

The strongest unadjusted tendency was versus `PC_VALIDATION_ONLY`
(raw p = 0.0202984; Holm p = 0.142089; median D = -0.00615196).
It is suggestive only, not confirmatory.

## Post-primary descriptive findings

Synthetic generator-level patterns are heterogeneous and descriptive only.

For the real descriptive anchor (`PC_NFPSO` vs `NF_BASE`), across 120 series:

- PC_NFPSO better: **39**
- exact ties: **9**
- NF_BASE better: **72**
- median D: **+0.0528447168**
- median MASE PC_NFPSO: **0.9491817702**
- median MASE NF_BASE: **0.9374341217**

This is descriptively in the direction of NF_BASE. No p-value or confidence
interval was computed, and this is not an eighth formal test.

## Claim boundaries

The ablation evidence does not justify equivalence, component necessity,
component dispensability, universal superiority, generator-level significance,
or any post hoc expansion of the seven-hypothesis family.

No raw model outcome, analysis unit, inferential statistic, or descriptive
statistic was recomputed by this final closure step.

The V2 ablation phase is closed and ready for manuscript integration.
