# V2 Protocol Amendment 005 — Canonical Synthetic Sampling and Contamination Clarifications

Status: LOCKED BEFORE DEVELOPMENT BENCHMARKING

## Purpose

The original Gate-2 freeze fixed the three canonical processes, their parameters, burn-in lengths, and the shared relative-Gaussian contamination mechanism, but it did not fully specify post-burn sample indexing or process-specific contamination RNG offsets. This amendment resolves those implementation details before any V2 development hyperparameter selection.

## Logistic map

Use `numpy.random.default_rng(dgp_seed)`. Draw one `x0 ~ Uniform(0.1,0.9)`. Apply exactly 500 map updates without recording. The first returned observation is the state after the 500th update. Each subsequent returned observation is separated by one additional map update.

The relative-observation-noise RNG is independently initialized with seed `dgp_seed + 7000`. The minimum relative-noise scale is 1.0.

## Henon map

Use `numpy.random.default_rng(dgp_seed)`. Draw `x0` and then `y0` as two consecutive independent `Uniform(-0.5,0.5)` draws. Apply exactly 1000 Henon updates without recording. The first returned observation is the x component after the 1000th update. Each subsequent returned observation is separated by one additional Henon update.

The relative-observation-noise RNG is independently initialized with seed `dgp_seed + 8000`. The minimum relative-noise scale is 1.0.

## Lorenz-63

Use `numpy.random.default_rng(dgp_seed)`. Draw initial conditions in the fixed order x, y, z from the already frozen ranges. Integrate with fixed-step classical RK4 using `dt=0.01`. Apply exactly 5000 RK4 steps without recording. The first returned observation is the x component immediately after the 5000th step. Each subsequent returned observation is separated by exactly 10 RK4 steps.

The relative-observation-noise RNG is independently initialized with seed `dgp_seed + 9000`. The minimum relative-noise scale is 1.0.

## Relative observation noise

For each process and DGP seed, one clean trajectory is generated and shared across all noise levels. Population standard deviation uses `ddof=0`.

For a fixed process and seed, every positive noise level reinitializes a separate contamination RNG with the same process-specific offset seed. Consequently the same standardized Gaussian vector is used across the positive noise levels and is scaled by the requested noise level. This preserves the already frozen dependence structure.

Noise level 0.0 returns an exact copy of the clean trajectory. Negative noise levels are invalid and must raise rather than being silently treated as noise-free.

## Six custom generators

The V1 source equations, innovation distributions, custom seed offsets 1000 through 6000, minimum scales, and the spiky-outbreak nonnegative projection agree with the frozen V2 custom specifications. V2 nevertheless uses an isolated implementation rather than importing the V1 module, so that V2 naming, input validation, finite checks, population-standard-deviation semantics, and seed firewall behavior are explicit.

## Timing

No V2 development or final forecasting outcome was generated or inspected to choose any rule in this amendment.
