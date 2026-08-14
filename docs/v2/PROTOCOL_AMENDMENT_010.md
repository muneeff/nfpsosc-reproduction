# Protocol Amendment V2-A010 — Ablation Execution and Reference Provenance

## Timing and scope

This clarification is frozen **after** the V2 external Final benchmark was
completed and closed at commit `ed1e713a869a0c0b62d098f6f15c664b1472f63e`, but **before any ablation
outcome is generated**.

It does not alter any ablation variant, selected hyperparameter, seed,
confirmatory noise level, metric, hypothesis, effect definition, confidence
interval, test, or multiplicity family.

Its purpose is to bind operational details that were not explicit in the
original locked ablation configuration.

## Exact PC-NFPSO reference reuse

The canonical external Final corpus is cryptographically identified by:

`393b7df1fca6d6dc2e57b60b06d6ccca7f15217e3a8c351b9427346829da299b`

A PC-NFPSO computation that is scientifically identical to an already completed
external Final task is **reused by exact task identity and immutable result
bytes**, rather than rerun.

For synthetic ablation, only optimizer seeds **41001, 41002, 41003** are used,
matching the frozen ablation design. This reuses **810** PC-NFPSO raw result
rows. Within process × DGP-seed × noise, the three MASE values are
median-aggregated; the three confirmatory noise aggregates are then averaged.

For the real anchor, the already frozen A009 rule is retained: PC-NFPSO is the
median across five optimizer seeds per real series. This reuses **600** raw
PC-NFPSO result rows.

Reuse is not retuning, re-selection, or outcome substitution.

## NF_BASE

NF_BASE is deterministic and receives one run per synthetic condition or real
series. It has no optimizer seed.

It uses:
- the same training-only frozen scaling path;
- subtractive-clustering antecedents with radius **1.0**;
- ridge alpha **0.05**;
- final consequent-only refit on the complete pre-test supervised window;
- fixed fitted parameters through test;
- observed true history to update one-step lag inputs.

## New execution ledger

Synthetic:
- 270 process × seed × noise conditions;
- NF_BASE: 270 new deterministic tasks;
- six non-reference stochastic variants × 3 optimizer seeds:
  **4860** new tasks;
- synthetic new total: **5130**.

Real:
- NF_BASE on 120 frozen real series: **120** new tasks;
- PC-NFPSO reference is reused.

Grand total of newly executed ablation tasks: **5250**.

Exact external PC-NFPSO raw result rows reused: **1410**
(810 synthetic + 600 real).

## Aggregation and failure handling

For each stochastic synthetic variant, all three optimizer replicates are
required before the within-condition median is defined. A survivor median is
not permitted.

All three confirmatory noise aggregates must be available for both sides of a
paired synthetic contrast. No imputation, substitution, or silent fallback is
allowed.

The seven pre-specified PC-NFPSO reference-contrast labels remain fixed in the
Holm family. Pair counts and failures are reported. If a contrast has no
estimable paired units, its raw p-value is fixed at 1.0 for multiplicity
accounting rather than deleting the hypothesis.

## Inferential scope

The **seven** confirmatory synthetic contrasts remain:

1. PC_NFPSO vs NF_BASE
2. PC_NFPSO vs NFPSO
3. PC_NFPSO vs P_NFPSO
4. PC_NFPSO vs C_NFPSO
5. PC_NFPSO vs PC_NO_SENSITIVITY
6. PC_NFPSO vs PC_NO_COVERAGE
7. PC_NFPSO vs PC_VALIDATION_ONLY

The real NF_BASE vs PC_NFPSO comparison is an external-domain **descriptive
anchor** and is not added to the seven-hypothesis synthetic Holm family.

Other factorial component contrasts remain descriptive unless separately
pre-specified for formal inference; they do not expand the frozen family.

## Execution firewall

Before the first new ablation outcome:
1. implementation and tests must be complete;
2. the exact ablation task manifest must be frozen;
3. the execution workspace identity must be frozen;
4. no external result may redefine variants, retune radius/alpha, change
   seeds/noise levels, or alter the inferential family.
