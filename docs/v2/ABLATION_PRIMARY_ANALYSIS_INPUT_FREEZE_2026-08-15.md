# V2 Ablation Primary Analysis Input Freeze

**Freeze date:** 2026-08-15

**Status:** `FROZEN_PRE_INFERENCE`

This artifact freezes the synthetic primary analysis input before any paired
effect, p-value, confidence interval, or Holm-adjusted result is computed.

## Source bindings

- Closed ablation result corpus SHA-256: `3628d71f833bd50ae9c5ff8799568e159bb89afa525e6d3bc5bafc77bc44a45b`
- Closed external Final aggregate SHA-256: `393b7df1fca6d6dc2e57b60b06d6ccca7f15217e3a8c351b9427346829da299b`
- Synthetic new-result rows used: **5,130**
- Frozen synthetic PC_NFPSO reference rows used: **810**

## Frozen primary analysis input

- Domain: **synthetic**
- Analysis unit: **process × DGP seed**
- Units: **90**
- Reference: **PC_NFPSO**
- Formal comparator variants: **7**
- Primary metric: **MASE**
- Optimizer aggregation: median over seeds **41001, 41002, 41003**
- Noise aggregation: arithmetic mean over **0.05, 0.10, 0.20**
- Unit identity SHA-256: `65cb8884d9f203d7b22781a1b2579b3b8a3584efcf8b3205c571e4b3c4b2e155`
- Unit table SHA-256: `40047d36b17e2bf495422e83f9730fb6d5e4f8ac9066435213fd667314379350`

## Reserved inferential contract

The already-frozen inferential contract is not executed here:

- `D = MASE_PC_NFPSO - MASE_variant`
- paired median effect
- exact 95% order-statistic median CI
- exact two-sided sign test
- Holm adjustment across exactly 7 formal contrasts
- familywise alpha = 0.05

No inferential statistic has been calculated by this freeze step.
