# V2 Protocol Amendment 003 — ETS Numerical Convergence

Status: LOCKED BEFORE DEVELOPMENT BENCHMARKING

## Scope
This amendment defines numerical validity for Holt-Winters ETS candidates during pre-test AICc selection.

It does not alter the ETS candidate grid, AICc selection criterion, Amendment V2-A001 tie-break rules, B1 fixed-parameter test-time policy, or any development/final split.

## Candidate validity
An optimized ETS candidate is eligible for AICc comparison only when fitting completes, all fitted parameters are finite, AICc is finite, optimizer results are available, and `mle_retvals.success` is exactly True.

An optimizer failure is treated as a failed candidate, logged with provenance, and excluded from model selection. No fallback or substituted forecast is permitted.

## Timing
This rule was frozen during implementation-level QC and before any V2 development or final benchmark was opened.
