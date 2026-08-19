# NFPSOSC V2 Project Provenance and Reproduction Guide

## Purpose

This document records the provenance, execution history, recovery actions, and reproducibility information for the NFPSOSC V2 Final benchmark execution.

## Final Execution Freeze History

Original freeze:

- Tag: `v2-final-execution-freeze-2026-08-14`
- Commit: `ebfbb758a41c0146c9da618c4ce873976e4b5289`
- Frozen tasks: 8780

## M4 Memory Failure

A Final execution failure occurred during M4 Monthly series loading.

Example:

- Source: M4
- Frequency: Monthly
- Series: M16885

Error:

```
MemoryError
Unable to allocate 1023 MiB
stage=load
```

## Root Cause

The previous loader path loaded the complete M4 group through `M4.load()`, causing excessive memory consumption.

This was an execution infrastructure problem only. The scientific protocol was not changed.

## Memory-Safe Correction

Commit:

`b50515c`

Message:

`fix(v2): load frozen M4 series individually to avoid memory failure`

A new loader:

`_load_m4_single_series()`

was introduced to load only the required frozen series.

## Data Integrity Check

Series:

`M16885`

```
OLD_LENGTH = 87
NEW_LENGTH = 87

OLD_HASH =
6418f27c40980e694b88ebc6cf92d90062286c8b69b8d8d82880c2264e1118c1

NEW_HASH =
6418f27c40980e694b88ebc6cf92d90062286c8b69b8d8d82880c2264e1118c1

MATCH = True
```

## New Execution Freeze

Tag:

`v2-final-memory-safe-execution-freeze-2026-08-16`

Commit:

`fb6465cbe82ea0cdcf3c57bb5b31ea36a804f270`

Purpose:

Execution-only recovery after deterministic M4 loading failure.

Scientific protocol unchanged.

## Commit Timeline

| Commit | Purpose |
|---|---|
| ebfbb75 | Original Final execution scheduler freeze |
| 88ab2e4 | Ignore external downloaded M4 dataset |
| b50515c | Memory-safe M4 single-series loader |
| fb6465c | Point Final execution gate to memory-safe freeze |

## Recovery Workspace

Original:

`outputs/v2/final_external`

Recovery:

`outputs/v2/final_external_memory_safe`

## Freeze Verification

Execution commit:

`fb6465cbe82ea0cdcf3c57bb5b31ea36a804f270`

Required tag:

`v2-final-memory-safe-execution-freeze-2026-08-16`

Manifest SHA256:

`763da2f1e18a1f2572e97f7ece8b17438db0e20c3fe04ac0d05980208f28d5fd`

Protocol fingerprint:

`545f5be0ec0d14b724ed6dd76189019ab78a72acb39fec0ed53878e2d67700c8`

Status:

`LOCKED_PRE_FINAL_OUTCOMES`

## Validation

After correction:

```
SUCCESSFUL_TOTAL = 21
FAILED_TOTAL = 0
REMAINING_TASKS = 8759
```

## Full Execution Command

```
python run_v2_final_parallel.py --workspace outputs\v2\final_external_memory_safe --workers 4
```

## Reviewer Note

The second freeze is an execution-only recovery. No benchmark definition, dataset selection, protocol parameter, or evaluation rule was changed after observing final outcomes.

# Final Paper Release Closure

## Final Reproducibility Release

Tag:

v2-final-paper-ready-2026-08-19

Commit:

a7dfdb4a4edcebb9d62a8bea8b0d043179736e5c

Repository:

https://github.com/muneeff/nfpsosc-reproduction/tree/v2-final-paper-ready-2026-08-19


## Final Scientific State

The final paper release was created after completion of:

- Final benchmark execution
- Statistical analysis
- External comparison analysis
- Ablation evaluation
- Manuscript table generation


## Relationship Between Execution Freeze and Paper Release

The memory-safe execution freeze:

v2-final-memory-safe-execution-freeze-2026-08-16

was an execution infrastructure correction only.

The final paper release:

v2-final-paper-ready-2026-08-19

contains the completed reproducibility artifacts, analysis scripts, generated tables, and manuscript-supporting outputs.

No benchmark definition, dataset selection, evaluation rule, hyperparameter selection procedure, or statistical protocol was modified after observing Final outcomes.


## Final Benchmark Summary

Total completed forecasting tasks:

8780

Synthetic tasks:

7140

Real tasks:

1640


## Final Reproducibility Status

Status:

LOCKED_FINAL_PAPER_RELEASE