# V2 Protocol Amendment 001 — AICc Tie-Breaks

Status: LOCKED BEFORE DEVELOPMENT BENCHMARKING

## Scope
This amendment resolves one implementation-level ambiguity in the frozen baseline protocol: the exact deterministic complexity-oriented tie-break for equal finite AICc values in SARIMA and ETS selection.

It does not alter candidate grids, AICc as the selection criterion, pre-test-only selection, failure handling, B1 test-time update policy, or any final-test rule.

## SARIMA
Primary criterion: minimum finite AICc.

Exact AICc ties are resolved by: smaller p+q+P+Q; smaller P+Q; smaller p+q; smaller d+D; smaller D; then lexicographic (p,d,q,P,D,Q,trend_code), where trend_code n=0 and c=1.

## ETS
Primary criterion: minimum finite AICc.

Exact AICc ties are resolved by: fewer active structural components; undamped before damped; nonseasonal before seasonal; no trend before additive trend; then deterministic lexicographic tuple.

## Timing
This amendment was created before any V2 development benchmark or final benchmark was opened.
