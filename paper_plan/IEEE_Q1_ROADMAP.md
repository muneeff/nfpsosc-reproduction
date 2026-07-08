# IEEE Q1 Roadmap for the SS-NFPSO Paper

Working target:

**IEEE Transactions on Fuzzy Systems**  
Alternative targets may include IEEE Transactions on Artificial Intelligence or IEEE Transactions on Emerging Topics in Computational Intelligence, depending on the final strength and framing.

---

## 1. Intended Positioning

The paper should be positioned as a theory-guided and diagnostically transparent study of sparse TSK neuro-fuzzy forecasting under data scarcity.

The paper should not be positioned as a simple application of NFPSO to one dataset.

Preferred framing:

> This study develops and evaluates a constrained sparse TSK neuro-fuzzy forecasting formulation trained by Projected Constricted PSO, focusing on output boundedness, input-output sensitivity, rule complexity, PSO dynamics, and leakage-aware generalization under data scarcity.

---

## 2. Target Journal Fit

## 2.1 IEEE Transactions on Fuzzy Systems

### Why it fits

- The core model is a TSK fuzzy inference system.
- The paper analyzes fuzzy-rule complexity.
- The contribution can be framed around interpretable, sparse, constrained fuzzy systems.
- Stability and sensitivity diagnostics are relevant to fuzzy-system design.

### What is required

- Clear fuzzy-system novelty.
- Formal mathematical results.
- Strong experimental evaluation.
- Careful comparison with relevant forecasting and fuzzy/neuro-fuzzy baselines.
- No overclaiming of universal superiority.

### Main risk

If the paper reads like a forecasting benchmark rather than a fuzzy-systems contribution, it may be rejected.

---

## 2.2 IEEE Transactions on Neural Networks and Learning Systems

### Why it may fit

- The paper involves learning systems, optimization, generalization, and nonlinear forecasting.

### Main challenge

The fuzzy-system contribution must be strong enough, and the learning-theoretic or optimization contribution must be substantial. This venue is very competitive.

---

## 2.3 IEEE Transactions on Artificial Intelligence

### Why it may fit

- The paper can be framed as interpretable AI for robust forecasting under data scarcity.

### Main challenge

The paper must appeal beyond fuzzy systems and provide broader AI relevance.

---

## 3. Required Manuscript Identity

The manuscript must clearly separate itself from earlier NFPSO-SC work.

Required statement:

> The present study does not claim novelty in the original combination of neuro-fuzzy forecasting and particle swarm optimization. Instead, it formulates the model within a constrained parameter domain and analyzes boundedness, sensitivity, rule complexity, PSO dynamics, and leakage-aware generalization.

This statement protects the manuscript from being seen as duplicate or incremental publication.

---

## 4. Minimum Scientific Contributions

The paper should contain at least five concrete contributions:

1. A constrained sparse TSK neuro-fuzzy forecasting formulation.
2. A Projected Constricted PSO training mechanism for bounded parameter search.
3. Conditional bounded-output and input-output sensitivity results.
4. Rule-complexity analysis linked to subtractive-clustering radius.
5. A theory-aligned experimental protocol measuring stability, sensitivity, complexity, PSO dynamics, and generalization.

Optional stronger contribution:

6. A public reproducibility package with leakage-aware implementation and experiment scripts.

---

## 5. Required Theoretical Components

## 5.1 Definitions

The paper must define:

- lagged input vector;
- TSK fuzzy rule;
- Gaussian membership function;
- firing strength;
- normalized rule weight;
- sparse rule base;
- parameter vector;
- constrained parameter domain;
- PSO position and velocity;
- projection operator;
- constriction or bounded update.

---

## 5.2 Proposition 1 — Output Boundedness

Required result:

If inputs and consequent parameters are bounded, and normalized rule weights are nonnegative and sum to at most one or approximately one with a positive denominator floor, then the TSK output is bounded.

Status:

- Required for Q1.
- Should be provable.

---

## 5.3 Proposition 2 — Input-Output Sensitivity Bound

Required result:

Under lower-bounded membership widths and bounded consequent parameters, the Jacobian norm or an upper bound on input-output sensitivity can be expressed in terms of rule count, width lower bound, consequent bounds, and input-domain bounds.

Status:

- Required but may be stated conditionally.
- Must avoid overclaiming global stability if the bound is loose.

---

## 5.4 Proposition 3 — Rule Complexity Control

Required result:

The number of selected subtractive-clustering centers is related to the clustering radius through a separation or covering argument under stated assumptions.

Status:

- Useful but should be cautious.
- If a strict proof is hard, state as a conditional complexity argument rather than a universal theorem.

---

## 5.5 Proposition 4 — Bounded PSO Dynamics

Required result:

Projection guarantees feasible particle positions remain inside the compact parameter domain. Constriction/bounded velocity assumptions help control update magnitudes.

Status:

- Required.
- Do not claim global convergence.

---

## 6. Required Experimental Components

## 6.1 Minimum Q1-Level Experimental Package

At minimum:

- 3–5 real datasets.
- 5+ synthetic controlled datasets.
- baseline comparison.
- radius sweep.
- noise sensitivity.
- seed robustness with at least 20 seeds.
- PSO budget sensitivity.
- stability diagnostics.
- statistical tests.
- failure-case analysis.

## 6.2 Stronger Package

Preferred:

- 10+ real time series.
- 6+ synthetic generators.
- 8–10 baselines.
- 30 random seeds.
- full radius sweep.
- full noise and PSO-budget sensitivity.
- public code repository.
- supplementary results.

---

## 7. Required Baselines

Minimum baselines:

- Naive-1.
- Seasonal Naive, if seasonality exists.
- Drift.
- ARIMA or Auto-ARIMA.
- ETS.
- Theta.
- SVR.
- XGBoost.
- Legacy NFPSO-SC.
- Proposed Projected Constricted NFPSO.

Optional but useful:

- ANFIS baseline, if implementation is reliable.
- Random forest regression.
- LightGBM, if used carefully.
- Simple average forecast combination.

---

## 8. Required Metrics

Main error metrics:

- MAE.
- RMSE.
- sMAPE.
- MASE.

Use MAPE cautiously, especially if zero or near-zero targets occur.

Diagnostics:

- Jacobian norm.
- Empirical Lipschitz estimate.
- Activation-sum statistics.
- Rule count.
- Parameter count.
- Parameter norm.
- Boundary hits.
- Mean velocity norm.
- Swarm diameter.
- Best objective.
- Runtime.
- Generalization gap.

---

## 9. Statistical Analysis

Required:

- Friedman test for multiple models.
- Holm-corrected post-hoc comparisons.
- Bootstrap confidence intervals.
- Seed-based variability summaries.

Optional:

- Diebold-Mariano or HAC-corrected paired forecast-error comparison, when assumptions are reasonable.

Do not rely only on average rank.

---

## 10. Reproducibility Requirements

The repository should contain:

- source code;
- experiment scripts;
- configuration files;
- data templates;
- instructions;
- no private raw data unless authorized;
- no output files revealing sensitive values;
- no local machine paths;
- no binary model files unless explicitly intended.

Current repository status:

- Basic source code exists.
- README exists.
- Data template exists.
- Outputs and private data are excluded.
- Additional Q1 experiment scripts still need to be added.

---

## 11. Manuscript Sections

Recommended structure:

1. Introduction
2. Relationship to Prior NFPSO-SC Work
3. Sparse TSK Neuro-Fuzzy Forecasting Model
4. Projected Constricted PSO Training
5. Stability and Complexity Analysis
6. Experimental Protocol
7. Results
8. Discussion
9. Threats to Validity
10. Conclusion

Alternative IEEE-style structure:

1. Introduction
2. Related Work
3. Proposed Method
4. Theoretical Analysis
5. Experimental Setup
6. Results and Discussion
7. Conclusion

The first structure is clearer for avoiding overlap with prior work.

---

## 12. Statements to Avoid

Avoid:

- “The proposed method always outperforms...”
- “PSO guarantees global optimum...”
- “The experiments prove stability...”
- “The model is universally robust...”
- “The rule count is always monotonic in radius...”
- “The legacy revenue dataset proves generalization...”

Use:

- “Under the stated assumptions...”
- “The diagnostics suggest...”
- “The results support...”
- “The method improves robustness in several settings...”
- “Failure cases indicate...”
- “The bound is conservative but informative...”

---

## 13. Before Submission Checklist

Before submitting to an IEEE Q1 journal, verify:

- [ ] The manuscript has a clear novelty statement.
- [ ] Prior NFPSO-SC work is acknowledged.
- [ ] The theory contains at least boundedness and sensitivity results.
- [ ] The rule-complexity claim is supported.
- [ ] The PSO-dynamics claim is not overstated.
- [ ] At least 3–5 real datasets are included.
- [ ] Synthetic controlled experiments are included.
- [ ] Baselines are strong and fair.
- [ ] No future-information leakage exists.
- [ ] Seed variability is reported.
- [ ] Runtime is reported.
- [ ] Statistical tests are included.
- [ ] Figures and tables directly support the claims.
- [ ] Code repository is clean.
- [ ] No private data or local paths are exposed.
- [ ] Authorship and acknowledgements are accurate.
- [ ] Declaration of competing interest is singular if there is one author.
- [ ] CRediT statement matches actual contribution.

---

## 14. Current Priority

Immediate next steps:

1. Add `paper_plan/` files.
2. Add `configs/` files.
3. Add synthetic data generators.
4. Add baseline models.
5. Run a minimal Q1 experiment package.
6. Review results before writing the final Results section.

Do not write strong claims until the minimal experiment package has been executed.
