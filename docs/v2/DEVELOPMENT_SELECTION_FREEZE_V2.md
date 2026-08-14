# V2 Development Selection Freeze

**Date:** 2026-08-14
**Status:** LOCKED after complete Development selection and before any Final benchmark execution.

## Preconditions

The frozen Development workspace was bound to:

- Code commit: `db4b18e74ae7c443fe7cbf837ba8664e63d21dca`
- Protocol fingerprint: `0b60efbdf693a7f2053c426806e6d44d5b146e53afb7615ca28398202478889e`
- Development manifest SHA-256: `0e9c8125228aaa3aaeee23d173c59d0c97bdf24ab71488a79aa8320b150a4f64`

All `28,350` planned Development runs completed successfully, with zero failed and zero pending runs.

## Cryptographic provenance

- Aggregate SHA-256 over the 28,350 Development result files:
  `8ff2e5d3d3861cec5d7d608610447b16f7905f8ecabfa74a90e057301c70246e`
- SHA-256 of the Development result-hash manifest:
  `5c89a8fbbd8eb2957a9a82b47d44c51e1974308b76b8b125b99f85ad6cae4e9d`
- SHA-256 of the frozen Development selection result:
  `f39ab0284b5058a8a7814006898b09ece39518ac1afb834fce590d88957675d6`

## Frozen selection rule

The Development candidate grid contained 35 `(radius, ridge_alpha)` pairs. For each candidate pair, the three optimizer replicates were aggregated by the median MASE within each Development condition. Candidate performance was then aggregated by the global median across all Development conditions. Exact ties were resolved by preferring the larger radius and then the larger ridge alpha.

No Final synthetic or Final real benchmark result was used for Development hyperparameter selection.

## Selected hyperparameters

- **Radius:** `1.0`
- **Ridge alpha:** `0.05`

These values are now locked for all subsequent Final benchmark runs. They must not be changed, retuned, or selectively overridden after inspection of Final benchmark outcomes.

## Interpretation

This file records an outcome-derived Development decision; it is not a protocol amendment and does not retroactively alter the frozen Development protocol. Its purpose is to bind the Final benchmark to the pre-Final selected hyperparameters and their provenance.
