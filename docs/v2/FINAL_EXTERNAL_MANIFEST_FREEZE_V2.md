# V2 Final External Manifest Freeze

Date: 2026-08-14

Status: LOCKED PRE-FINAL OUTCOME EXECUTION

Stage: 3F-2D

## Bound implementation

- Git commit: `c56504bc8d95ebac9aa6e51c165769903077f348`
- Protocol fingerprint: `545f5be0ec0d14b724ed6dd76189019ab78a72acb39fec0ed53878e2d67700c8`

## Frozen Final external manifest

- Path: `outputs/v2/final_external/final_external_manifest.json`
- SHA-256: `763da2f1e18a1f2572e97f7ece8b17438db0e20c3fe04ac0d05980208f28d5fd`
- The `.sha256` sidecar matched the manifest hash at freeze time.

## Frozen task design

- Total tasks: 8,780
- Synthetic tasks: 7,140
  - PC-NFPSO: 2,700
  - Baselines: 4,440
- Real tasks: 1,640
  - PC-NFPSO: 600
  - Baselines: 1,040
- Frozen Final real series: 120
- Final DGP seeds: 20
- Final optimizer seeds: 5

## Frozen Development-selected hyperparameters

- Radius: `1.0`
- Ridge alpha: `0.05`

## Run-ID bounds

- First run ID: `34d165df0541ec1cfadd03c9b123545df778e37371630f500de78876c6937715`
- Last run ID: `459621514788e0e8963cbab4022949c01854d2e10872f1e8f91e4ec04f19bd5d`

## Outcome state at freeze

`FINAL_OUTCOME_EXECUTION_ENABLED=False`

`NO_FINAL_OUTCOMES_GENERATED=True`

No Final forecast, Final accuracy metric, or Final performance outcome had been generated when this manifest was created and frozen.

## Lock rule

The frozen Final external manifest is the sole admissible task design for subsequent Final external benchmark execution. After Final outcome execution begins, task identities, selected hyperparameters, Final seed sets, frozen real-series identities, method-support rules, and protocol fingerprint must not be altered in response to observed Final results.
