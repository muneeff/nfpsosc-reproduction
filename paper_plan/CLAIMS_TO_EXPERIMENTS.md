# Claims-to-Experiments Map for the SS-NFPSO Paper

Working title:

**Stability, Complexity, and Generalization Analysis of Sparse TSK Neuro-Fuzzy Forecasting Trained by Projected Constricted Particle Swarm Optimization**

This document maps each scientific claim in the paper to the theoretical result, empirical diagnostic, table, and figure required to support it. The goal is to prevent unsupported claims and to keep the manuscript suitable for a high-quality IEEE submission.

---

## 1. Positioning of the Paper

The paper does **not** claim that the basic combination of neuro-fuzzy forecasting and Particle Swarm Optimization is new. Earlier NFPSO-SC work is treated as prior empirical work.

The new contribution is the constrained, diagnostically transparent formulation of sparse TSK neuro-fuzzy forecasting, trained by Projected Constricted PSO, with emphasis on:

- bounded output behavior;
- input-output sensitivity;
- rule-complexity control;
- bounded PSO dynamics;
- leakage-aware empirical evaluation;
- robustness under data scarcity.

---

## 2. Claim 1 — Bounded Output

### Claim

Under bounded inputs, bounded consequent parameters, positive lower-bounded membership widths, and normalized fuzzy-rule weights, the sparse TSK neuro-fuzzy forecaster produces bounded outputs.

### Theoretical support required

- Define the input domain.
- Define bounds on consequent coefficients and intercepts.
- Define the normalized firing-strength aggregation.
- Prove that the output is a convex or normalized weighted combination of bounded local consequents.
- State the bound explicitly.

### Empirical diagnostics

Required diagnostics:

- maximum absolute prediction;
- maximum absolute local consequent output;
- parameter L2 norm;
- minimum and median activation sum;
- boundary-hit count;
- actual-vs-predicted plot;
- residual range.

### Tables and figures

- **Table 4:** Stability diagnostics.
- **Figure 3:** Actual vs predicted.
- **Figure 5:** Swarm and parameter dynamics.
- Optional appendix figure: prediction range vs theoretical output bound.

### Risk if omitted

If boundedness is claimed without an explicit proposition and diagnostics, reviewers may treat it as vague stability language rather than a real contribution.

---

## 3. Claim 2 — Input-Output Sensitivity Control

### Claim

Constraining membership widths and consequent parameters allows the input-output sensitivity of the TSK forecaster to be bounded or at least diagnostically monitored.

### Theoretical support required

- Derive partial derivatives of Gaussian membership functions.
- Derive or upper-bound the Jacobian norm of the normalized TSK output.
- State dependence on:
  - number of rules;
  - minimum membership width;
  - maximum consequent coefficient magnitude;
  - input-domain bound;
  - activation denominator floor.

### Empirical diagnostics

Required diagnostics:

- mean Jacobian norm;
- maximum Jacobian norm;
- empirical Lipschitz mean;
- empirical Lipschitz maximum;
- perturbation sensitivity under controlled noise;
- noise-level vs error increase.

### Tables and figures

- **Table 4:** Jacobian and empirical Lipschitz diagnostics.
- **Figure 7:** Noise sensitivity curves.
- **Figure 6:** Radius vs Jacobian norm.
- Optional appendix: theoretical vs empirical sensitivity ratio.

### Risk if omitted

The paper may appear to use “stability” as a metaphor rather than a measurable or analyzable property.

---

## 4. Claim 3 — Rule-Complexity Control

### Claim

The subtractive-clustering radius controls the number of fuzzy rules and thus the number of model parameters, creating a measurable tradeoff between sparsity, accuracy, sensitivity, and runtime.

### Theoretical support required

- Define the rule-generation mechanism.
- State the relationship between cluster radius and rule density.
- Provide a conditional complexity argument using separation, covering, or packing intuition.
- Avoid claiming a strict monotonic theorem unless the implemented algorithm guarantees it.

### Empirical diagnostics

Required diagnostics:

- number of selected rules;
- number of trainable parameters;
- center separation;
- runtime;
- train/validation/test error;
- generalization gap;
- Jacobian norm;
- empirical Lipschitz.

### Tables and figures

- **Table 5:** Radius sweep results.
- **Figure 6:** Radius vs rules, RMSE, Jacobian, and runtime.
- **Figure 9:** Complexity-generalization tradeoff.

### Risk if omitted

If the word “sparse” appears in the title but the rule-complexity tradeoff is not analyzed, the paper will look incomplete.

---

## 5. Claim 4 — Bounded PSO Dynamics

### Claim

Projection keeps particle positions inside a compact parameter domain, while constriction reduces uncontrolled growth of velocities. Together they make PSO dynamics more diagnostically stable, although they do not guarantee global optimality.

### Theoretical support required

- Define the projected update:
  - unconstrained PSO update;
  - projection onto a compact feasible domain.
- State that projection guarantees feasible particle positions.
- Discuss constriction or bounded velocity assumptions.
- Avoid claiming global convergence for nonconvex NFPSO training.

### Empirical diagnostics

Required diagnostics:

- best objective curve;
- mean velocity norm;
- swarm diameter;
- best parameter drift;
- boundary hits;
- parameter norm;
- per-seed objective variability.

### Tables and figures

- **Table 7:** PSO budget sensitivity.
- **Figure 4:** Best objective vs iteration.
- **Figure 5:** Swarm diameter, velocity norm, boundary hits, and parameter drift.
- **Figure 8:** Seed robustness.

### Risk if omitted

Reviewers may object that PSO is heuristic and that the paper overstates training stability.

---

## 6. Claim 5 — Robustness and Generalization Under Data Scarcity

### Claim

The constrained sparse NFPSO formulation can improve robustness and reduce unstable behavior under some data-scarce conditions, but it does not guarantee universal superiority across all time-series domains.

### Theoretical support required

- Explain why fewer rules and bounded parameters may reduce variance.
- Do not claim universal forecasting dominance.
- Connect complexity, sensitivity, and generalization gap cautiously.

### Empirical diagnostics

Required diagnostics:

- train error;
- validation error;
- test error;
- generalization gap;
- seed mean ± standard deviation;
- average rank across datasets and metrics;
- failure-case analysis;
- statistical tests.

### Tables and figures

- **Table 3:** Main forecasting performance.
- **Table 6:** Seed robustness.
- **Table 8:** Statistical tests.
- **Figure 8:** Seed boxplots.
- **Figure 9:** Complexity vs generalization gap.

### Risk if omitted

The paper may be seen as another empirical forecasting comparison rather than a theory-guided analysis.

---

## 7. Required Experiments by Claim

| Experiment | Claim 1 | Claim 2 | Claim 3 | Claim 4 | Claim 5 |
|---|---:|---:|---:|---:|---:|
| Legacy reproduction | ✓ | partial | partial | partial | partial |
| Research baseline | ✓ | ✓ | partial | ✓ | ✓ |
| Radius sweep | partial | ✓ | ✓ | partial | ✓ |
| Noise sensitivity | partial | ✓ | partial | partial | ✓ |
| Seed study | partial | ✓ | partial | ✓ | ✓ |
| PSO budget sensitivity | partial | partial | partial | ✓ | ✓ |
| Baseline comparison | partial | partial | partial | partial | ✓ |
| Synthetic benchmarks | ✓ | ✓ | ✓ | partial | ✓ |
| Real dataset panel | partial | partial | partial | partial | ✓ |
| Failure-case analysis | partial | ✓ | ✓ | ✓ | ✓ |

---

## 8. Minimum Evidence Required Before Writing Results

Before writing the final Results section, the following outputs should exist locally:

- `outputs/q1_minimal/panel/metrics_long.csv`
- `outputs/q1_minimal/panel/diagnostics_long.csv`
- `outputs/q1_minimal/radius_sweep/radius_sweep_results.csv`
- `outputs/q1_minimal/noise_sensitivity/noise_sensitivity_results.csv`
- `outputs/q1_minimal/seed_study/seed_study_results.csv`
- `outputs/q1_minimal/pso_budget/pso_budget_results.csv`
- `outputs/q1_minimal/statistics/friedman_results.csv`
- `outputs/q1_minimal/statistics/wilcoxon_holm_results.csv`
- `outputs/q1_minimal/paper_tables/`
- `outputs/q1_minimal/paper_figures/`

These output files should not be committed to the public repository if they contain real data, predictions, or sensitive values.

---

## 9. Claims That Must Not Be Made

Do not claim:

- NFPSO is universally superior.
- PSO reaches the global optimum.
- Experiments prove the theorems.
- Subtractive clustering always gives a monotonic rule count for every implementation and dataset.
- Bounded diagnostics imply forecasting accuracy.
- One legacy dataset is sufficient for broad generalization.

Acceptable wording:

- “The results support...”
- “The diagnostics indicate...”
- “Under the stated assumptions...”
- “The constrained formulation enables...”
- “The method improves robustness on several but not all datasets...”
