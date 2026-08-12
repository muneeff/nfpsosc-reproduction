# V2 Protocol Amendment 006 — Lorenz-63 RK4 Floating-Point Evaluation Order

Status: LOCKED BEFORE DEVELOPMENT BENCHMARKING

## Why this amendment is necessary

The frozen canonical benchmark already specifies Lorenz-63, classical fixed-step RK4, `dt=0.01`, a 5000-step burn-in, and sampling every 10 integration steps. During pre-development implementation testing, an additional reproducibility issue was identified: Lorenz-63 is chaotic, so algebraically equivalent floating-point rearrangements of the RK4 update can produce materially different trajectories after a long burn-in.

No V2 development or final forecasting outcome was generated or inspected.

## Exact state representation

Use NumPy float64 arrays for the three-dimensional state.

## Right-hand side

Evaluate the derivatives in this source-level form:

- `dx = sigma*(y-x)`
- `dy = x*(rho-z)-y`
- `dz = x*y-beta*z`

with the already frozen parameters `sigma=10`, `rho=28`, and `beta=8/3`.

## Exact RK4 step

For each integration step:

```text
k1 = f(state)
k2 = f(state + 0.5*dt*k1)
k3 = f(state + 0.5*dt*k2)
k4 = f(state + dt*k3)
state_next = state + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
```

with `dt=0.01`.

The final update must use the parenthesization above. It must not be rewritten as `dt*(...)/6.0`, and no adaptive or external ODE solver may replace the fixed-step implementation.

## Burn-in and sampling

Apply exactly 5000 frozen RK4 steps before recording. The first returned observation is the x component immediately after the 5000th step. Apply exactly 10 additional RK4 steps between consecutive returned observations.

## Reproducibility scope

This freezes the source-level evaluation order for the pinned V2 Python/NumPy environment. It is not a claim of bit-for-bit identity across arbitrary future numerical libraries, compilers, or hardware.

## Timing

This clarification is frozen before any V2 development hyperparameter-selection run or final benchmark run.
