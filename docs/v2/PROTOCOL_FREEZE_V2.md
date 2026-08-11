# PC-NFPSO V2 Experimental Protocol

Status: LOCKED - Gate 2 protocol specification complete. No V2 forecasting outcome was generated before this freeze.
Date: 2026-08-11
V1 parent commit: fdd9ed5bf5ba1be23660aee1e2a785eb1cdb2476

## Locked decision A1 - Objective family
V2 will use an explicitly defined composite optimization objective. The objective contains fitting error, chronological validation error, a sensitivity term, and an activation-quality penalty. The exact coefficients are not yet frozen and must be fixed using development data only before the final benchmark is opened.

## Locked decision B1 - Forecast update policy
Model parameters are fitted before the test window and remain fixed throughout final evaluation. Each one-step-ahead forecast may use only observations that are genuinely available at that forecast origin. Machine-learning and neuro-fuzzy parameters are not re-estimated at each origin. State-space methods may update their latent state with newly observed values but may not re-estimate model parameters during the test window.

## Locked decision C - Leakage firewall
The experimental lifecycle is strictly Development -> Freeze -> Final locked benchmark. Development data may be used for method design and hyperparameter selection. Final benchmark observations and final performance values may not be used for selecting radius, lag count, ridge strength, objective weights, PSO parameters, baseline hyperparameters, or any other modeling choice.

If a genuine software defect is discovered after the final benchmark has been opened, the affected final evaluation is invalidated. The defect is repaired and tested on development data, and a new untouched final benchmark must be used. The original final results remain archived.

## Feature definition
The base V2 PC-NFPSO model uses raw chronological lag vectors. Engineered means, standard deviations, differences, or other summary features are not part of the base model unless they are implemented and declared as a separate experimental variant or ablation.

## Failure policy
Silent fallback is prohibited. A failed Theta, ARIMA, ETS, SVR, XGBoost, or other model forecast must never be scored under the original model name using another forecasting rule. Failures must be explicitly recorded. Mandatory models must pass development smoke tests before the final configuration is frozen.

## Current prohibition
No V2 benchmark execution is permitted until Gate 2 is completed.

## Gate 2 readiness

All pre-specified protocol components required before implementation work are locked. The pending-before-Gate-2 list is empty. Implementation changes and development experiments may begin only after the cryptographic protocol-freeze commit and tag are created.
## Locked decision 2A - Exact composite objective

The V2 PC-NFPSO antecedent search minimizes:

J = 0.5*RMSE_fit + 1.0*RMSE_validation + 0.001*S + 0.01*A.

All quantities are evaluated in the training-fitted scaled domain. S is the mean squared L2 norm of the analytic input Jacobian over the union of fitting and chronological-validation samples. A is a low-fuzzy-coverage penalty evaluated on chronological-validation samples. For validation sample i, let a_i denote the sum of its raw Gaussian rule firing strengths. Then A is the mean of max(0, log(1e-4) - log(max(a_i,1e-300))) squared.

The coefficients 0.5, 1.0, 0.001, and 0.01 and the activation floor 1e-4 are fixed inherited method constants. They will not be tuned on V2 development or final data. Their contribution will instead be assessed through pre-specified ablation variants. This prevents performance-driven selection of objective weights.

Terminology note: the fourth term is called a low-coverage penalty, not a generic activation-quality penalty, because it penalizes insufficient total raw firing strength.

## Locked decision 2B - Ridge regularization policy

Ridge regularization is treated as a development-stage hyperparameter rather than as a value selected from final-test performance. The pre-specified candidate set is:

{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1}.

One global alpha will be selected for the base PC-NFPSO model by minimizing the median MASE across all pre-specified development forecasting conditions. Final benchmark observations or final performance values are prohibited from this selection. If the primary selection statistic is numerically tied, the larger alpha is selected as a deterministic stability-oriented tie-break.

The selected alpha is then frozen and used across the final benchmark; no final-dataset-specific ridge retuning is permitted. After antecedent optimization, TSK consequent coefficients are refitted on the complete pre-test training window using this frozen alpha.

The candidate set deliberately contains both 1e-4, which was used by the V1 implementation, and 5e-2 (0.05), which was stated in the rejected manuscript. Neither historical value receives preferential treatment.

The selection statistic is MASE rather than pooled raw RMSE because the benchmark contains series with heterogeneous scales.

## Locked decision 2C - Lag policy

The base PC-NFPSO uses a contiguous raw chronological lag vector:

x_t = [y_(t-L), ..., y_(t-1)].

The lag dimension is determined without performance-based model selection. Let m denote a known seasonal period supplied by frozen dataset metadata or by the synthetic data-generating specification. If no seasonal period is known, m is treated as 1. The deterministic rule is:

L = min(12, max(5, m)).

Thus nonseasonal and yearly series use L=5, quarterly series use L=5, and monthly series with m=12 use L=12. Seasonal periods larger than 12 are capped at L=12 to preserve the low-dimensional design of the sparse-data forecaster.

No development or final performance value is used to choose L for the base model, and no final-series-specific lag retuning is permitted. Each one-step-ahead input contains only observations genuinely available before its forecast origin.

Supervised lag-based baselines such as SVR and XGBoost receive the same L assigned to the corresponding series. ARIMA, ETS, and Theta are not artificially converted to the lag-matrix representation; any seasonal period used by those methods must nevertheless be obtained from frozen metadata rather than inferred from final-test performance.

Alternative lag dimensions may later be evaluated only as explicitly pre-specified sensitivity or ablation analyses; their results cannot be used to redefine the frozen base PC-NFPSO after the final benchmark is opened.

## Locked decision 2D - Projected-constricted PSO numerical specification

PC-NFPSO optimizes only the Gaussian antecedent centers and positive widths. TSK consequent coefficients are not swarm coordinates; they are solved by ridge regression for each candidate antecedent geometry.

The swarm budget is fixed at 12 particles and 40 iterations with no performance-based early stopping. These are inherited fixed computational-budget constants and are not tuned on V2 development or final results.

Constriction uses phi1 = 2.05 and phi2 = 2.05, giving phi = 4.10. The Clerc-style constriction coefficient is

chi = 2 / (phi - 2 + sqrt(phi^2 - 4*phi)),

which is approximately 0.729843788. In constricted mode the coefficients are fixed as w = chi, c1 = chi*phi1, and c2 = chi*phi2, giving c1 = c2 approximately 1.496179766. Inertia damping is not applied in constricted mode.

Initial velocities are zero. Particle 0 is the subtractive-clustering antecedent vector clipped to the feasible box, while all remaining particles are sampled uniformly from that box using numpy.random.default_rng with a pre-specified run seed. Independent random vectors r1 and r2 are generated for every particle update across all dimensions. Particle updates are sequential, so an improved global best can influence later particles in the same iteration.

For each feature j, let range_j = max(max(X_fit,j)-min(X_fit,j), 1e-3). Center bounds are [min_j - 0.05*range_j, max_j + 0.05*range_j]. Gaussian-width bounds are [0.05*range_j, 1.00*range_j]. All bounds are derived exclusively from X_fit, not validation or test observations.

Velocity is clipped coordinate-wise to plus or minus 0.10 of the corresponding feasible parameter span. If a proposed position exceeds a bound, the corresponding velocity component is sign-reversed and the proposed position is projected/clipped onto the feasible box.

A non-finite initial objective value is a fatal run error. A later non-finite candidate cost is assigned the maximum representable floating-point value and cannot become a personal or global best. Every run executes exactly 40 iterations.

The allocation of random seeds across development conditions, optimizer replicates, and final conditions is specified separately in the locked seed protocol.

## Locked decision 2E - Synthetic development/final seed firewall

The synthetic benchmark is divided into disjoint development and final seed domains. V1 seeds 1 through 5 have already been observed during the rejected study and are therefore classified as legacy seeds. They may be used only for debugging and regression testing and are prohibited from V2 hyperparameter selection and final statistical inference.

Development DGP seeds are 1001 through 1010 inclusive. Development PC-NFPSO runs use optimizer seeds 31001, 31002, and 31003. All algorithm design, radius selection, ridge selection, and any other permitted model-selection activity must occur exclusively within the development domain.

Final DGP seeds are 2001 through 2020 inclusive. Final PC-NFPSO runs use optimizer seeds 41001 through 41005. Final outcomes may not be generated, inspected, or used until the complete Gate-2 protocol has been committed and tagged.

Optimizer replicates do not constitute independent benchmark observations. For a stochastic method, replicate results are first aggregated within each generator/DGP-seed/noise condition using the median. Statistical comparisons are then conducted at the condition or higher pre-specified block level rather than treating optimizer repeats as additional sample size.

The same optimizer-seed set is used across PC-NFPSO and stochastic ablation variants within a protocol layer to support paired comparisons and reduce Monte-Carlo noise.

Noise-level conditions generated from the same generator and DGP seed are recognized as dependent. The generator/DGP-seed pair is therefore recorded as a dependence block for the later inferential protocol.

Knowing the pre-registered final seed integers is not itself considered access to final outcomes; generating, examining, or adapting decisions to those outcomes before the protocol freeze is prohibited.

The exact synthetic DGP equations, generator parameters, noise grid, and benchmark composition remain to be frozen before Gate 2 can pass.

## Locked decision 2F-1 - Custom synthetic data-generating processes

The V2 custom stress-test suite contains six fully specified generators: linear AR, nonlinear sine, regime switching, spiky outbreak, noisy seasonal, and short-memory. Their exact equations and numerical parameters are stored in configs/v2/synthetic_dgps_v2.json.

All custom confirmatory series have length 180, an 80/20 chronological train/test split, and one-step-ahead forecasting. The confirmatory contamination levels are 0.05, 0.10, and 0.20. The clean noise-free case (0.0) is retained only as a descriptive diagnostic and is not treated as replicated confirmatory evidence.

This restriction is necessary because nonlinear_sine and noisy_seasonal have deterministic clean signals: changing the DGP seed at noise_level=0 does not create independent series. Counting those duplicate trajectories as independent observations would constitute pseudo-replication.

Relative contamination is Gaussian. For a clean generated trajectory y, the contamination standard deviation is noise_level times max(population_std(y), minimum_scale). Generator-specific minimum scales and seed offsets are frozen in the manifest. For stochastic generators this contamination is additional to their intrinsic innovations.

The custom generators are stress tests, not claimed to constitute a recognized canonical benchmark collection. A separate canonical synthetic benchmark suite remains mandatory before Gate 2 can pass, specifically to address the reviewer request for well-known synthetic benchmarks.

## Locked decision 2F-2 - Canonical nonlinear/chaotic benchmark processes

The custom synthetic suite is supplemented by three classical nonlinear/chaotic processes: the logistic map, the Henon map, and the Lorenz-63 system. These are described as canonical benchmark processes and are not claimed to constitute a single formally established benchmark collection.

The logistic benchmark uses x_(t+1)=4*x_t*(1-x_t). Each DGP seed defines x_0 by a Uniform(0.1,0.9) draw, followed by 500 discarded burn-in iterations. The observed univariate series is x.

The Henon benchmark uses x_(t+1)=1-1.4*x_t^2+y_t and y_(t+1)=0.3*x_t. Each seed generates x_0 and y_0 independently from Uniform(-0.5,0.5); 1000 iterations are discarded before recording the x coordinate.

Lorenz-63 uses sigma=10, rho=28, and beta=8/3. V2 numerically integrates the three-dimensional system using deterministic fixed-step classical RK4 with dt=0.01. Five thousand integration steps are discarded as burn-in. The x coordinate is then recorded every 10 integration steps, producing an effective sampling interval of 0.1. Seed-specific initial conditions are generated independently as x_0~Uniform(-15,15), y_0~Uniform(-20,20), and z_0~Uniform(5,35).

The burn-in lengths, seed-based initial-condition distributions, numerical integration rule, and sampling intervals are V2 experimental design choices fixed before benchmark execution; they are not represented as requirements of the original source papers.

Each recorded series contains 180 observations and uses the frozen 80/20 chronological split and one-step-ahead forecasting protocol. Confirmatory relative observation-noise levels are 0.05, 0.10, and 0.20. The zero-noise condition is retained for descriptive diagnostics only.

For each process and DGP seed, one clean trajectory is generated first and the noise-level variants derive from that trajectory. Consequently the noise-level variants are dependent and share the process/DGP-seed dependence block. The already frozen development and final seed domains remain strictly disjoint.

These canonical processes supplement the six custom stress-test generators, giving V2 nine synthetic process families before consideration of real-world series.

## Locked decision 2G-1 - Real-series eligibility and deterministic partition policy

The confirmatory real-data benchmark is drawn from the public M3 and M4 collections using Yearly, Quarterly, and Monthly strata. The V2 study does not manually choose individual series and does not rank candidates according to forecasting performance.

Eligible series must contain 40 through 180 observations inclusive, contain only finite observations with no internal missing values, and support the frozen chronological 80/20 train/test design. After applying the frequency-dependent frozen lag rule, at least 20 supervised training rows and at least 8 test observations must remain. The MASE scaling denominator, computed exclusively from the pre-test training segment using the known seasonal period, must exceed 1e-12.

Within each of the six source/frequency strata, eligible series are sorted by the hexadecimal SHA-256 digest of the fixed string pc-nfpso-v2|source|frequency|series_id. The first 10 eligible IDs are assigned to development and the next 20 to the locked final benchmark. Thus the target public real-data benchmark contains 60 development series and 120 final series. There is no RNG-based or manual outcome-based selection.

The five public series previously used in V1 (M4 Y6688, Q6355, M6356 and M3 Y404, Q687) are explicitly excluded from both V2 development and V2 final confirmatory evidence. The Yemen tax-revenue series is retained only as a legacy external descriptive case study because both the series and its V1 forecasting results have already been examined. It does not contribute to V2 hyperparameter selection or confirmatory statistical inference.

The public real-development subset is reserved for baseline development, smoke testing, implementation validation, and exploratory robustness checks. PC-NFPSO structural hyperparameters such as clustering radius and ridge strength are selected only within the synthetic development firewall, preventing real final-series characteristics from feeding back into the proposed model design.

Minimal model-independent inspection required to establish eligibility, lengths, and deterministic IDs is permitted before Gate 2. Forecasts, forecast errors, rankings, or algorithm-specific diagnostic outcomes for the final IDs are prohibited until Gate 2 has been committed and tagged.

The exact source-file SHA-256 hashes and the exact 60 development plus 120 final series IDs must be materialized into a frozen manifest before Gate 2 can pass.

## Locked decision 2G-2 - Exact real-series manifest

The deterministic real-series selector produced exactly 180 public series: 60 development series and 120 final-locked series, balanced as 10 development and 20 final series in each of the six M3/M4 frequency strata. No V1 public series appears in the V2 manifest, and the development/final intersection is empty.

The authoritative JSON manifest SHA-256 is 4e36b9bf6e46719c5e9274ae523e82dc31aee1b7355e2abb188eaf70f7665602. The corresponding CSV SHA-256 is 7d893bfe897becdabb4cea1c98ad2bfea9cbc00d7e14df8f60ad56fe0ed582fe. The frozen loader is datasetsforecast 1.0.1. These identifiers define the real-data benchmark independently of all forecasting outcomes.

## Locked decision 2H-0 - Subtractive-clustering radius policy

The candidate subtractive-clustering radii are fixed before any V2 forecasting experiment as {0.25, 0.35, 0.55, 0.75, 1.00}. One global radius is selected using synthetic development conditions only. Legacy V1 seeds and all V2 final seeds are prohibited from radius selection.

For each radius, stochastic optimizer replicates are first aggregated within each development condition by the median. The global radius minimizing median development MASE is selected and then frozen. If the primary selection statistic is numerically tied, the larger radius is chosen as the pre-specified deterministic tie-break. No generator-specific, real-series-specific, or final-dataset-specific radius selection is permitted.

Using the historical candidate grid does not transfer V1's historical selection result into V2; V2 re-selects the radius exclusively inside the new disjoint development firewall.

## Locked decision 2H - Baseline models, tuning, and failure handling

V2 evaluates nine comparator families: one-step naive, seasonal naive when m>1, drift, SARIMA, ETS, Theta, lagged ridge regression, RBF-SVR, and XGBoost. The exact candidate spaces and deterministic selection rules are frozen in configs/v2/baselines_v2.json.

No baseline is permitted to silently substitute another forecasting method after an exception. Every candidate failure and forecast failure is recorded explicitly. If all candidates for a baseline fail on a condition, that baseline-condition pair is marked failed rather than replaced with a naive forecast.

Seasonal naive is defined only for m>1. It is not emitted as a separate comparator when m=1 because in that case it is algebraically identical to the one-step naive method.

SARIMA and ETS use frozen model-selection rules based solely on the pre-test training segment. SARIMA searches a bounded low-order seasonal/nonseasonal space and chooses the finite-AICc minimum. ETS searches additive trend/damping and, when m>1, optional additive seasonality. Multiplicative ETS components are excluded to avoid changing eligibility according to positivity of the observed series.

Theta has no V2 hyperparameter search. Seasonal periods come from frozen metadata. Seasonal series are deseasonalized using the known period with the seasonality test disabled and additive decomposition. Forecast extraction is positional. In particular, pandas label indexing with forecast[0] is prohibited because it caused the V1 Theta implementation to fail and silently fall back to naive predictions.

Ridge, RBF-SVR, and XGBoost use the same frozen lag dimension as PC-NFPSO. Their candidate configurations are selected using a chronological validation tail contained entirely inside the pre-test training window. The validation size is V=min(max(5,ceil(0.20*N_train)), N_train-L-10), with V required to be at least five. After selection, the chosen model and all preprocessing are refitted on the complete pre-test training window and then held fixed throughout the test window.

For supervised models, validation forecasts may use observed lag values that become available at each validation origin, but model parameters are not re-estimated. Test-time behavior follows the same information policy.

Before any final benchmark execution, every mandatory baseline must pass a development QC suite checking forecast length, finite outputs, failure logging, prediction provenance, and absence of forecast substitution. Theta receives a dedicated regression test using a nonzero pandas index. Final execution is blocked if the QC gate fails.

## Locked decision 2I - Primary metric and inferential estimand

The primary accuracy metric is MASE. Its scaling denominator is computed exclusively from the pre-test training segment using the frozen seasonal period m; m=1 is used for nonseasonal series. Final test observations never enter the MASE scaling denominator.

For each comparator b, the paired effect is D_i = MASE_PC-NFPSO,i - MASE_b,i. Negative values favor PC-NFPSO. The primary inferential estimand is the population median paired MASE difference, estimated by the sample median of D_i.

The primary hypothesis test is a two-sided paired sign test. Exact zero differences are recorded as ties and excluded from the binomial trial count. The corresponding primary effect interval is a two-sided 95 percent nonparametric order-statistic confidence interval for the median paired difference. Thus the reported primary effect and its confidence interval target the same location quantity rather than mixing a rank test with a confidence interval for an arithmetic mean.

Wilcoxon signed-rank results may be reported only as sensitivity analyses and are not used for the primary confirmatory claim.

For synthetic experiments, optimizer replicates are first aggregated by the median within each process/DGP-seed/noise condition. The three confirmatory noise levels 0.05, 0.10, and 0.20 are then averaged within each process/DGP-seed block. Consequently the final synthetic confirmatory analysis has a target of 9 processes times 20 DGP seeds = 180 analysis blocks. Optimizer replicates and noise variants are not treated as independent sample-size inflation. The zero-noise diagnostic condition is excluded from confirmatory inference.

For real data, one final series is one analysis unit, yielding 120 target units across six balanced source/frequency strata. Synthetic and real domains are analyzed and reported separately; they are not collapsed into a single artificial overall benchmark score.

MAE, RMSE, sMAPE, and RMSSE are secondary measures. Raw RMSE and MAE may be reported by process, dataset, or stratum but are not pooled across heterogeneous scales as the primary ranking statistic.

Model failure rates are always reported independently of forecast accuracy. Silent fallback remains prohibited. If final failures occur, accuracy comparisons use only explicitly identified successful pairs, state the resulting pair count, and are described as conditional on successful execution; failure cases cannot silently disappear from the interpretation.

## Locked decision 2J - Complete statistical analysis plan

The primary confirmatory multiplicity family consists of all nine PC-NFPSO-versus-baseline comparisons in each of the two confirmatory domains, synthetic and real, for a maximum family size of 18 hypotheses. Holm step-down adjustment controls the family-wise Type-I error rate at 0.05 across this full family rather than separately within individual result tables.

For each hypothesis, D is defined as MASE_PC-NFPSO minus MASE_baseline. The primary two-sided exact sign test counts D<0 as a PC-NFPSO win and D>0 as a loss. Exact-zero differences are reported as ties and excluded from the binomial trial count. If all differences are ties, the raw p-value is fixed at 1.0.

The primary point effect is the sample median of finite paired MASE differences. Its 95 percent confidence interval is obtained by exact nonparametric binomial inversion of the ordered paired differences. Thus the primary effect, interval, and sign-based hypothesis test form one coherent robust inferential analysis rather than mixing a signed-rank p-value with a confidence interval for an arithmetic mean.

A directional claim that PC-NFPSO is statistically superior requires all of: Holm-adjusted p<0.05, a negative sample median D, and more wins than losses. A significant result in the opposite direction is reported as evidence favoring the comparator. Non-significance is reported as insufficient evidence of superiority and is never described as equivalence.

The complete Holm family is fixed before final evaluation. A comparator test that cannot be estimated because of execution failure is assigned p=1.0 for multiplicity accounting rather than deleted from the family after results are known. Forecast failure counts and rates are always reported separately from conditional accuracy comparisons.

Synthetic inference operates on process/DGP-seed units after median aggregation across optimizer replicates and arithmetic-mean aggregation across the three confirmatory noise levels. Real inference operates on individual final series. Synthetic and real observations are not pooled into a single hypothesis test.

Wilcoxon signed-rank analyses are sensitivity analyses only. Arithmetic mean paired differences with stratified 10,000-resample bootstrap intervals are likewise secondary sensitivity summaries. For synthetic data the bootstrap resamples DGP-seed units within process; for real data it resamples series within source/frequency stratum. The fixed bootstrap seed is 90210.

## Locked decision 2K - Internal ablation matrix

The primary structural ablation consists of NF_BASE, NFPSO, P_NFPSO, C_NFPSO, and PC_NFPSO. NF_BASE uses subtractive-clustering antecedents and frozen-ridge consequents without PSO. The remaining four form a two-by-two optimizer design crossing nonconstricted versus constricted swarm dynamics with feasible-rejection versus projected boundary handling.

Projection is explicitly separated from constriction in V2. Under projected handling, offending velocity components are reflected and the proposed coordinates are clipped to the feasible box. Under nonprojected feasible-rejection handling, an out-of-box proposal is rejected, the previous particle position is retained, and offending velocity components are reflected. Infeasible Gaussian widths or centers are never evaluated.

All stochastic structural variants receive identical lag rules, clustering radius, ridge alpha, objective, parameter bounds, swarm size, iteration budget, initialization protocol, and paired optimizer seeds. Therefore P_NFPSO versus NFPSO isolates the projection operator under nonconstricted dynamics, PC_NFPSO versus C_NFPSO isolates projection under constricted dynamics, C_NFPSO versus NFPSO evaluates constriction without projection, and PC_NFPSO versus P_NFPSO evaluates constriction with projection.

Three objective ablations are also fixed: PC_NO_SENSITIVITY sets the sensitivity weight to zero; PC_NO_COVERAGE removes the low-coverage penalty; and PC_VALIDATION_ONLY minimizes chronological validation RMSE alone. These variants are not retuned independently.

The controlled synthetic ablation uses all nine process families, final DGP seeds 2001 through 2010, noise levels 0.05, 0.10, and 0.20, and optimizer seeds 41001 through 41003. Optimizer repeats are median-aggregated and noise levels are averaged before inference, yielding 90 process/DGP-seed analysis units.

The seven pre-specified PC_NFPSO-versus-ablation comparisons constitute a separate internal Holm family at family-wise alpha 0.05. The effect, confidence interval, and sign test follow the same frozen MASE/median-difference framework used by the external analysis.

On the 120 real final series, only the computationally economical NF_BASE versus PC_NFPSO anchor ablation is required. The full optimizer factorial is reserved for the controlled synthetic domain.

No engineered-feature ablation is included in the primary matrix because engineered features are not part of the frozen V2 base-model definition. A zero-ridge ablation is likewise excluded because ridge estimation is treated as a frozen estimation component rather than as a claimed algorithmic novelty.
