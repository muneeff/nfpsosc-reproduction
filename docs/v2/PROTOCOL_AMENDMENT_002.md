# V2 Protocol Amendment 002 — SARIMA Numerical Convergence

Status: LOCKED BEFORE DEVELOPMENT BENCHMARKING

## Scope
This amendment defines numerical validity for SARIMA candidates during pre-test AICc selection.

It does not alter the candidate grid, AICc selection criterion, Amendment V2-A001 tie-break rules, B1 test-time update policy, or any development/final split.

## Candidate validity
A SARIMA candidate is eligible for AICc comparison only when fitting completes, all fitted parameters are finite, AICc is finite, and `mle_retvals["converged"]` is exactly True.

A non-converged candidate is treated as a failed candidate, logged with provenance, and excluded from model selection. No fallback or substituted forecast is permitted.

## Timing
This rule was frozen after implementation-level QC exposed a non-converged toy candidate and before any V2 development or final benchmark was opened.
