# V2 Protocol Amendment 004 — TSK Implementation Clarifications

Status: LOCKED BEFORE DEVELOPMENT BENCHMARKING

## Purpose
This amendment resolves implementation details that were not fully specified in the original V2 freeze. No development or final performance result was used.

## Scaling
PC-NFPSO uses affine min-max scaling to [0.1, 0.9], inherited from the V1 neuro-fuzzy implementation. The scaler is estimated exclusively from raw observations available through the final fitting target. Chronological-validation targets and all test observations are excluded. The same scaler remains frozen during validation, antecedent optimization, complete-pretest consequent refitting, and test forecasting.

## Subtractive-clustering initialization
Subtractive clustering is applied deterministically to the joint scaled matrix [X_fit, y_fit]. Accepted cluster centers are restricted to their input dimensions to initialize Gaussian antecedent centers. For feature j, range_j=max(ptp(X_fit[:,j]),1e-3), and the initial Gaussian width is radius*range_j/sqrt(8). Particle 0 is the resulting antecedent vector clipped to the frozen feasible box.

## Ridge solve
All first-order TSK consequent coefficients, including intercepts, are regularized. The solve is (Phi.T Phi + alpha I) beta = Phi.T y. There is no hidden default alpha; the frozen selected alpha must be passed explicitly.

## Coupled radius-alpha selection
Because radius and ridge alpha jointly affect forecasting performance, V2 evaluates the Cartesian product of their already frozen candidate grids on synthetic development conditions. Optimizer replicates are first aggregated by median MASE within each development forecasting condition. The global selection statistic is the median of those condition values. Lower is better. Exact ties choose the larger radius and then the larger alpha. No final or real-series retuning is permitted.

## Numerical firing implementation
Raw Gaussian log firing is -0.5 times the summed squared standardized distance. Normalized rule weights are computed with a stable log-sum-exp softmax. The low-coverage penalty is evaluated directly from the log total firing strength using the frozen 1e-300 numerical floor; this is algebraically equivalent to the already frozen objective while avoiding underflow. The analytic input Jacobian must use the same normalized-weight definition and pass finite-difference tests.

## Timing
This amendment is frozen before any V2 development hyperparameter selection or final benchmark execution.
