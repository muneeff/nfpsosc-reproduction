# Figures and Tables Plan for the SS-NFPSO Paper

Working title:

**Stability, Complexity, and Generalization Analysis of Sparse TSK Neuro-Fuzzy Forecasting Trained by Projected Constricted Particle Swarm Optimization**

This document defines the figures and tables required for the manuscript. Each table or figure must support a specific claim. Avoid decorative figures that do not serve the argument.

---

## 1. Table Plan

## Table 1 — Dataset Summary

### Purpose

Describe the real and synthetic datasets used in the evaluation.

### Columns

| Column | Description |
|---|---|
| Dataset | Dataset name |
| Type | Real or synthetic |
| Domain | Economics, epidemiology, energy, synthetic, etc. |
| Frequency | Monthly, weekly, daily, synthetic step |
| Length | Number of observations |
| Train | Number of training observations |
| Validation | Number of validation observations, if separate |
| Test | Number of test observations |
| Seasonality | Yes/No/Approximate |
| Role | Legacy case, benchmark, stress test, synthetic control |
| Source | Public source or local authorized source |

### Notes

- The Yemeni tax-revenue series should be described as a legacy or motivating case.
- If the raw data are not public, clearly state that only a template is provided in the repository.

---

## Table 2 — Model and Hyperparameter Configuration

### Purpose

Make the experimental setup reproducible.

### Columns

| Column | Description |
|---|---|
| Model | Naive, ARIMA, ETS, Theta, SVR, XGBoost, legacy NFPSO, constrained NFPSO |
| Key parameters | Main model settings |
| Selection method | Fixed, validation-selected, grid, default, etc. |
| Leakage control | How future information is avoided |
| Repeats/seeds | Number of random seeds |
| Implementation | Python package or custom implementation |

### Notes

- Report PSO particles, iterations, constriction, projection bounds, radius, lag length, and penalty weights.
- Avoid reporting only final tuned values without explaining how they were chosen.

---

## Table 3 — Main Forecasting Performance

### Purpose

Compare forecasting accuracy across datasets and methods.

### Columns

| Dataset | Model | MAE | RMSE | MAPE | sMAPE | MASE | Rank |
|---|---|---:|---:|---:|---:|---:|---:|

### Notes

- Use mean ± standard deviation if multiple seeds are used.
- If MAPE is invalid due to zero or near-zero values, use sMAPE and MASE as primary metrics.
- Do not rank using R² if error metrics are the main target.

---

## Table 4 — Stability and Sensitivity Diagnostics

### Purpose

Connect the theoretical stability discussion with empirical diagnostics.

### Columns

| Dataset | Method | Rules | Parameters | Jacobian mean | Jacobian max | Empirical Lipschitz mean | Empirical Lipschitz max | Min activation sum | Parameter norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

### Notes

- This table is essential for a theory-guided paper.
- It should not be replaced by performance metrics alone.

---

## Table 5 — Radius Sweep

### Purpose

Show the relationship between subtractive-clustering radius, rule count, complexity, error, and sensitivity.

### Columns

| Dataset | Radius | Rules | Parameters | MAE | RMSE | sMAPE | Jacobian max | Empirical Lipschitz max | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

### Notes

- Include enough radius values to show a real pattern.
- Avoid over-interpreting nonmonotonic behavior.

---

## Table 6 — Seed Robustness

### Purpose

Show that PSO results are not based on a single lucky seed.

### Columns

| Dataset | Method | Seeds | MAE mean ± std | RMSE mean ± std | sMAPE mean ± std | Jacobian max mean ± std | Runtime mean ± std |
|---|---|---:|---:|---:|---:|---:|---:|

### Notes

- Minimum recommended seeds: 20.
- Stronger version: 30 seeds.

---

## Table 7 — PSO Budget Sensitivity

### Purpose

Show the effect of particles and iterations on accuracy, stability, and runtime.

### Columns

| Dataset | Particles | Iterations | MAE | RMSE | sMAPE | Best objective | Boundary hits | Swarm diameter final | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

### Notes

- Helps justify whether a small PSO budget is acceptable.
- If higher budget improves results substantially, do not hide it.

---

## Table 8 — Statistical Testing

### Purpose

Assess whether observed performance differences are statistically meaningful.

### Columns

| Comparison | Test | Statistic | p-value | Correction | Adjusted p-value | Significant? |
|---|---|---:|---:|---|---:|---|

### Suggested tests

- Friedman test across multiple models and datasets.
- Holm-corrected Wilcoxon signed-rank tests.
- Bootstrap confidence intervals for paired error differences.
- Diebold-Mariano or HAC-corrected comparisons only when appropriate for serial forecast errors.

---

# 2. Figure Plan

## Figure 1 — Overall Framework

### Purpose

Show the full SS-NFPSO forecasting pipeline.

### Content

```text
Time-series input
→ Lag-window construction
→ Subtractive clustering
→ Sparse TSK fuzzy-rule base
→ Projected constricted PSO training
→ Forecast
→ Diagnostics: sensitivity, complexity, PSO dynamics, generalization gap
```

### Notes

- This figure should be clean and IEEE-style.
- Avoid decorative icons.

---

## Figure 2 — Theory-to-Empirical Evidence Map

### Purpose

Show how each theoretical claim is evaluated empirically.

### Content

```text
Bounded parameters → bounded output diagnostics
Minimum sigma + bounded consequents → Jacobian / Lipschitz diagnostics
SC radius → rule complexity diagnostics
Projection/constriction → PSO dynamics diagnostics
Complexity + sensitivity → generalization gap
```

### Notes

- This figure helps reviewers see the paper logic.

---

## Figure 3 — Actual vs Predicted

### Purpose

Show forecasting behavior on representative datasets.

### Content

- Observed values.
- Constrained NFPSO predictions.
- Optionally best baseline predictions.

### Notes

- Use only representative datasets in the main paper.
- Put additional datasets in appendix or supplementary material.
- Avoid showing private sensitive values publicly unless authorized.

---

## Figure 4 — PSO Best Objective Curve

### Purpose

Show optimization behavior.

### Content

- Best objective vs iteration.
- Multiple seeds or shaded mean ± standard deviation if available.

### Notes

- A single smooth curve is weaker than seed-averaged behavior.

---

## Figure 5 — Swarm Dynamics

### Purpose

Show bounded PSO behavior.

### Recommended panels

- Mean velocity norm vs iteration.
- Swarm diameter vs iteration.
- Best parameter drift vs iteration.
- Boundary hits vs iteration.

### Notes

- This figure directly supports the projected/constricted PSO claim.

---

## Figure 6 — Radius Sweep

### Purpose

Show rule-complexity tradeoff.

### Recommended panels

- Radius vs number of rules.
- Radius vs RMSE or sMAPE.
- Radius vs Jacobian max or empirical Lipschitz.
- Radius vs runtime.

### Notes

- This is one of the most important figures for the “sparse” and “complexity-aware” claims.

---

## Figure 7 — Noise Sensitivity

### Purpose

Show robustness to input perturbations.

### Recommended panels

- Noise level vs RMSE.
- Noise level vs empirical Lipschitz.
- Noise level vs prediction variance.

### Notes

- Use synthetic and real datasets if possible.
- For real datasets, explain how noise is added without corrupting training/test causality.

---

## Figure 8 — Seed Robustness

### Purpose

Show variability across random seeds.

### Recommended panels

- Boxplot of RMSE across seeds.
- Boxplot of Jacobian max across seeds.
- Boxplot of best objective across seeds.

### Notes

- Essential because PSO is stochastic.

---

## Figure 9 — Complexity-Generalization Tradeoff

### Purpose

Connect rule count, sensitivity, and generalization.

### Recommended panels

- Rules vs train-test gap.
- Parameter count vs test error.
- Jacobian max vs generalization gap.
- Runtime vs error.

### Notes

- This figure should support a cautious conclusion, not a universal law.

---

# 3. Appendix / Supplementary Material

Suggested supplementary items:

- Full per-dataset results.
- Full radius sweep results.
- Full seed results.
- Full hyperparameter table.
- Additional actual-vs-predicted figures.
- Failure-case plots.
- Detailed proof steps if the target journal prefers concise main text.

---

# 4. Figure Quality Requirements for IEEE Submission

- Use vector graphics where possible: PDF, EPS, or SVG.
- Use readable labels at journal-column size.
- Avoid bright decorative colors.
- Use consistent font sizes.
- Use consistent metric names.
- Do not crowd too many panels into one figure.
- Every figure must be cited and discussed in the text.
- Do not include figures that are not used in the argument.

---

# 5. Table Quality Requirements

- Use consistent decimal precision.
- Bold best values only if it does not create misleading emphasis.
- Report mean ± standard deviation when stochastic methods are used.
- State whether lower or higher is better.
- Explain negative R² if used.
- Avoid too many metrics in one table if readability suffers.

---

# 6. Minimum Main-Paper Set

If the paper must be concise, keep the following in the main manuscript:

- Table 1 — Dataset summary.
- Table 3 — Main forecasting performance.
- Table 4 — Stability diagnostics.
- Table 5 — Radius sweep.
- Table 6 — Seed robustness.
- Figure 1 — Framework.
- Figure 5 — Swarm dynamics.
- Figure 6 — Radius sweep.
- Figure 7 — Noise sensitivity.
- Figure 9 — Complexity-generalization tradeoff.

Move the rest to appendix/supplementary material.
