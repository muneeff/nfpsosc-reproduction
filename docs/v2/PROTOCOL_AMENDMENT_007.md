# V2 Protocol Amendment 007 — PC-NFPSO Validation Size and Development Failure Semantics

Status: LOCKED BEFORE DEVELOPMENT BENCHMARKING

## Purpose

The Gate-2 freeze defines a fitting segment followed by chronological validation for PC-NFPSO, but does not explicitly state the size of that validation tail. The validation formula frozen for supervised comparator tuning is baseline-specific and must not be silently reused for PC-NFPSO.

This amendment also freezes how top-level development-run failures affect the global radius-alpha selection. No V2 development or final forecasting outcome has been generated or inspected to choose these rules.

## PC-NFPSO chronological validation

For a pre-test raw series and frozen lag dimension `L`, construct the frozen contiguous supervised lag representation. Let:

`N = len(raw_pretest) - L`

be the number of pre-test supervised rows.

The PC-NFPSO validation size is:

`V = max(8, floor(0.20*N))`

Require at least 11 fitting rows:

`N - V >= 11`

The last `V` pre-test supervised rows are chronological validation. All earlier supervised rows are fitting.

This is the PC-NFPSO method-validation rule. It is distinct from the comparator inner-tuning formula already frozen in `baselines_v2.json`.

The rule inherits the leakage-aware V1 PC-NFPSO research-baseline split and is now made explicit before V2 development.

## Scaling

Amendment A004 remains unchanged. The PC-NFPSO scaler is estimated only from raw observations available through the final fitting target. The same scaler is frozen for validation, antecedent optimization, complete-pretest consequent refitting, and test forecasting.

## Synthetic development split

Every V2 synthetic series has length 180. The frozen 80/20 split therefore means:

- first 144 observations: pre-test training;
- final 36 observations: test.

The lag rule remains:

`L = min(12, max(5, m))`

using the frozen seasonal period `m`.

For synthetic development, this yields validation size 27 when `L=5` and 26 when `L=12`.

## Development MASE

The MASE denominator is computed only from the raw pre-test training window using the frozen seasonal period `m`:

`mean(abs(y_train[t] - y_train[t-m]))`

The denominator must exceed `1e-12`. Test observations never enter the denominator.

## Test update policy

PC-NFPSO parameters remain fixed throughout the test window. At each one-step origin the forecast uses the latest observed history; after forecasting, the actual test observation is appended for the next origin. Predictions are never fed back as observed history.

## Development run failures

Top-level development-run failures are not accuracy values.

Therefore:

- no silent fallback;
- no forecast substitution;
- no arbitrary MASE penalty;
- no failed-run deletion from a radius-alpha pair;
- every failure records deterministic run ID, failure stage, exception type, and message;
- global radius-alpha selection is blocked unless every expected development run has succeeded with a finite non-negative MASE.

For the frozen development design, the required successful score count is 28,350.

## Timing

These rules are frozen before any V2 development hyperparameter-selection outcome or final benchmark outcome is generated.
