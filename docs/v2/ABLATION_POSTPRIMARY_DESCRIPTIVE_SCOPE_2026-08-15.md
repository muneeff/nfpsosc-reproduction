# V2 Post-Primary Ablation Descriptive Scope

**Freeze date:** 2026-08-15

**Status:** `POST_PRIMARY_DESCRIPTIVE_SCOPE_FROZEN`

The seven-hypothesis primary ablation family is already closed. None of its
seven contrasts was Holm-significant. The analyses below are explicitly
post-primary and descriptive; they cannot redefine that confirmatory result.

## Synthetic descriptive scope

Source: the frozen 90-unit synthetic MASE table.

For the seven frozen PC_NFPSO reference contrasts:

- report median unit-level MASE by variant;
- reuse the already archived overall paired win/tie/loss counts and median D;
- for each of the 9 generators, report n=10 units, PC-better count, exact ties,
  comparator-better count, and median D;
- run no generator-level p-values, confidence intervals, or multiplicity tests.

## Real descriptive anchor

Only PC_NFPSO versus NF_BASE is permitted.

- 120 frozen real series;
- PC_NFPSO MASE: median across optimizer seeds 41001-41005 within each series;
- NF_BASE: one deterministic result per series;
- report paired series count, PC-better/tie/NF_BASE-better counts, median D,
  median PC_NFPSO MASE, and median NF_BASE MASE;
- no p-value;
- no confidence interval;
- this is not an eighth formal test.

## Explicit exclusions

No new hypothesis tests, ablation Wilcoxon tests, ablation bootstrap,
secondary-metric inference, real-domain factorial tests, unfrozen pairwise
tests between nonreference variants, equivalence tests, or component
necessity/dispensability claims are permitted by this descriptive scope.

The confirmatory ablation conclusion remains: **0 of 7 contrasts were
Holm-significant**.
