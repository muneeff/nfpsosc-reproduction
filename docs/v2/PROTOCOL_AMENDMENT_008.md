# V2 Protocol Amendment 008 — Hénon Basin-Valid Initial-Condition Rejection Rule

Status: **LOCKED BEFORE DEVELOPMENT BENCHMARKING**

## Trigger

A numerical-feasibility audit was run before any V2 development model fitting, forecast metric calculation, hyperparameter selection, or final benchmark evaluation.

The audit checked clean DGP generation only. All 90 development process/seed combinations were finite. One frozen final combination failed: Hénon map with DGP seed 2003 produced a non-finite trajectory under the original single-draw initial-condition rule. Debug seed 1 also failed.

This is therefore a DGP-definition defect, not a forecasting-performance result.

## Unchanged Hénon definition

The following remain unchanged:

- `a = 1.4`
- `b = 0.3`
- candidate `x0` and `y0` are independently drawn from `Uniform(-0.5, 0.5)`
- RNG is `numpy.random.default_rng(dgp_seed)`
- burn-in is exactly 1000 Hénon updates
- the first returned observation is the x component after update 1000
- 180 x observations are returned with stride 1
- relative observation-noise seed offset remains 8000
- relative-noise minimum scale remains 1.0

## Deterministic rejection rule

For each Hénon DGP seed:

1. Initialize exactly one `numpy.random.default_rng(dgp_seed)`.
2. Draw one candidate pair in fixed order: x first, then y.
3. Test that candidate for the complete state horizon needed to generate the frozen series:
   - 1000 burn-in updates;
   - 179 further updates between the 180 returned observations;
   - total feasibility horizon = 1179 Hénon updates.
4. A candidate is feasible only if, throughout that horizon, both state components are finite and satisfy:
   - `abs(x) <= 1e6`
   - `abs(y) <= 1e6`
5. If the candidate is infeasible, draw the next x,y pair from the same RNG and test again.
6. Accept the first feasible candidate.
7. If no feasible candidate is found within 10,000 candidate pairs, raise a fatal `RuntimeError`. No alternate seed, clipped trajectory, or fallback series is permitted.
8. After accepting a candidate, restart the recurrence from that accepted pair and generate the frozen burn-in and returned trajectory exactly as specified.

The `1e6` bound is a numerical escape guard, not a redefinition of the Hénon attractor.

## Effect on frozen seed domains

The pre-amendment feasibility audit found:

- Development seeds 1001–1010: all 10 first candidate pairs are feasible. Their generated clean Hénon trajectories remain unchanged.
- Final seeds 2001–2020: seed 2003 is the only failed first-candidate case.
- Under this deterministic rejection rule, seed 2003 accepts candidate pair 3:
  - `x0 = 0.48212728096900015`
  - `y0 = -0.3580221771573131`
- Debug seed 1 also accepts candidate pair 3:
  - `x0 = -0.18816854798951455`
  - `y0 = -0.07667355102742435`

No development or final seed is replaced.

## Performance blindness

This amendment was specified using numerical DGP feasibility only. No PC-NFPSO or baseline model was fitted in the feasibility audit; no MASE, RMSE, MAE, sMAPE, RMSSE, statistical test, or hyperparameter-selection outcome was inspected.

## Existing development workspace

The previously created development workspace is bound to the pre-A008 protocol fingerprint. It must be retained as historical provenance but is **invalid for execution after A008**.

After A008 is committed and the Hénon implementation/tests pass, a new zero-outcome development workspace must be created and frozen against the new code commit and protocol fingerprint.
