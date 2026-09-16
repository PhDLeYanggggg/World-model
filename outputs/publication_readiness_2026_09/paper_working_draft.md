# When to Trust Neural Motion Forecasts: Baseline-Relative Joint Intervention for Multi-Agent Forecasting

Working draft, 2026-09-16. Method proposal with a completed, negative three-seed
development experiment. No independent confirmation or submission-ready claim.

## Abstract

Multi-agent motion predictors are commonly evaluated by average trajectory error,
yet an improved average can conceal degradation on motion already well predicted
by a simple baseline. We investigate baseline-relative selective intervention:
learning benefit and harm from out-of-fold predictions and selecting neural
replacements over an observed interaction graph. An initial eight-observed,
twelve-predicted-step development study uses three training seeds and physically
grouped crossfit folds. All seeds select the causal constant-velocity floor;
uncontrolled neural forecasts worsen the prespecified normalized ADE by
7.09--7.90%. Joint and independent controls do not differ at the evaluated
policies. We identify strong sensitivity to small past normalization scales,
without treating a favorable alternative metric as confirmation. The method's
claimed advantage remains unestablished. Strong public predictors, matched-count
and deferral controls, broader independent scenes and confirmatory evaluation
are required before a positive submission claim.

## 1. Introduction

Strong causal predictors can be difficult to improve consistently. A neural predictor may capture turns or interactions while performing worse on straight, low-variance motion. Replacing the baseline everywhere therefore asks a different question from deciding where the neural model has useful information. The latter question is especially relevant under scene shift, where a global average provides limited guidance about the risk of a local intervention.

Multi-agent prediction introduces another difficulty. Per-agent decisions can combine incompatible futures even if each predictor produces a coherent scene when used alone. We propose to treat the intervention vector itself as a structured prediction and to evaluate its excess error relative to a fixed baseline.

The study tests whether baseline-relative loss supervision and joint intervention improve mixed forecasts beyond cost-aware routing alone, and whether that effect survives dependence-aware evaluation. Regression deferral and joint compatibility already have substantial prior work. The contribution cannot be established by renaming those components; it requires matched controls, a defensible methodological distinction and new independent results.

## 2. Related Work

AgentFormer jointly models social and temporal structure with agent-aware attention (Yuan et al., ICCV 2021). EqMotion introduces equivariant motion prediction and invariant interaction reasoning (Xu et al., CVPR 2023). SingularTrajectory studies a unified representation across trajectory-prediction tasks (Bae et al., CVPR 2024). These works preclude claiming that temporal attention, relative geometry or a common motion representation are novel by themselves.

Joint Metrics Matter studies joint forecasting errors and collisions (Weng et al., 2023). The present proposal must demonstrate a benefit beyond adding a joint metric or training penalty: it concerns which predictions from competing predictors can be selected together.

Cost-sensitive expert deferral predates this proposal ([Mozannar and Sontag, ICML 2020](https://proceedings.mlr.press/v119/mozannar20b.html)). [Mao, Mohri and Zhong (ICML 2024)](https://proceedings.mlr.press/v235/mao24d.html) explicitly study regression deferral with a fixed predictor. Our cost heads therefore require comparison to established deferral objectives, not only confidence thresholds. A two-action adaptation is now implemented and synthetically tested; its real comparison is pending.

Selective-regression work also shows that reduced coverage need not protect every subgroup ([Shah et al., ICML 2022](https://proceedings.mlr.press/v162/shah22a.html)). Our easy-error slice is a different construct, but the warning motivates reporting slice-specific damage rather than treating reduced intervention as a guarantee. The [source-scoped review](joint_intervention/deferral_and_coverage_prior_work.md) distinguishes these established results from our remaining hypotheses.

Conformal Risk Control and Learn then Test provide established tools for controlling losses or selecting risk-constrained policies. SODA-MPC combines conformalized OOD monitoring with reachability fallback in control. We target excess forecasting loss relative to a fixed predictor; we do not claim formal physical safety from low ADE.

[HCP](https://arxiv.org/html/2306.06342v4) requires both across-group and within-group exchangeability. [GHCP](https://arxiv.org/html/2608.15500v1) uses initial target-group labels under additional sampling assumptions; it is not a general repair for serially dependent windows. [CAFHT](https://arxiv.org/html/2402.09623v2) permits dependence within trajectories but requires exchangeability across trajectories for simultaneous path coverage. [Adaptive CP for motion planning](https://proceedings.mlr.press/v211/dixit23a/dixit23a.pdf) studies online coverage and feasible control. None of these guarantees alone orders a selected point forecast against our baseline. The [assumption audit and falsifiable claim matrix](joint_intervention/statistical_assumptions_and_claims.md) separate these tasks and specify which comparisons remain missing.

## 3. Method

Use the observed agent histories, legal past-only context and candidate rollouts to estimate gain and harm. Train these heads using cross-fitted predictions within training recordings. Freeze both the floor and candidate predictor before training/calibrating the deployment gate. The gate minimizes estimated excess loss over agents plus an interaction penalty on the mixed forecast, subject to supported risk budgets.

A past-context Transformer encodes observed history and neighbor tokens; queries
contain requested prediction times and baseline rollouts, never ground-truth
future coordinates or validity masks. The completed v1 experiment uses normalized
coordinate MSE and a fixed update budget. For error functional L, baseline B,
candidate N and target Y, define g = L(B,Y) - L(N,Y), benefit = max(g,0), and
harm = max(-g,0). Paths are normalized using observed context only. Separate
nonnegative costs imply predicted gain = benefit - harm, an internally coherent
prediction, not calibrated safety. Scene-held-fold predictions supply cost
targets; every producer and fitted ancestor must exclude the whole held fold.
Ridge and neural heads use the same 11,966 OOF examples and fit-only preprocessing.
Neither has established a predictive advantage in the first real experiment.

The [backend checks](supervised_backend/implementation_and_limits.md) verify
synthetic CPU/MPS recovery and real-input invariance under future corruption.
The [new real experiment](8to12_development_v1/results.md) supplies separate
development evidence. Matched realized-count intervention, public forecaster
comparison and independent risk calibration remain pending.

The [neural cost-head path](neural_cost_head/implementation_and_limits.md) now also fits nonnegative benefit/harm regression on the same verified OOF inputs as the ridge control, with fit-only normalization and unchanged candidate forecasts. This enables a capacity-controlled comparison rather than attributing gains to an untrained interface. Synthetic CPU/MPS recovery is verified; its development fixture selects the floor, and apparent intervention occurs only on queries lacking complete labels. Neither a real neural advantage nor calibrated safety follows from this implementation.

The [development evaluator](development_evaluation/implementation_and_limits.md)
compares floor, uncontrolled, budget-constrained unary, scene-uniform and
pairwise-joint controls on identical candidate forecasts. Scene membership uses
past observations, not future-label completeness. Raw errors remain recording-local;
the current normalized summary equally weights physical scenes. Common budget
caps do not establish equal realized coverage. The first real comparison is now
complete and negative; matched-count mechanism evidence is still pending.

A [frozen-policy calibration interface](risk_calibration/implementation_and_limits.md) now consumes those exports without refitting. It requires a prespecified family, order, risk functional and scene aggregation. Missing-label interventions receive the worst bounded loss instead of being excluded. For bounded [0,1] scene losses, its current reference screen adds sqrt(log(M*K/delta)/(2*n)) to the empirical mean for M policies and K risks, using n physical-scene clusters. An unchanged baseline has analytically zero excess risk; a learned policy that happens not to intervene on the observed sample still requires a sampling bound. These are conditional statistical statements, not evidence that the actual scenes are IID, a novel risk theorem, a 2% relative easy-error guarantee, or physical safety. Only synthetic integration has been executed; no real calibration result is available.

A binary optimizer solves this decision with MILP, using past-only coordinate
restoration, baseline-relative pair costs and explicit predicted-harm/intervention
budgets. Its real v1 development comparison shows no gain over independent
selection. The [implementation specification](joint_intervention/method_and_checks.md)
separates this mechanism from the statistical assumptions. JFP already studies
unary/pairwise forecast compatibility and heuristic overlap penalties; joint
optimization alone is not a novelty claim. Its proposed value still depends on
matched comparisons and independent evaluation. See the
[focused related-work audit](joint_intervention/related_work_constraints.md).

An additional [exact-count control](matched_coverage/method_and_limits.md) sets the joint intervention count to the independent policy's count on each observed query, before labels are read. It retains the same forecasts, support and predicted-harm cap. This isolates a change in selected identities from a change in coverage, conditional on the reference rule; it does not match realized risk. Forced-count outputs may be worse than the baseline and are diagnostic, not deployment policies. Solver failures remain unmatched in the ledger, and zero-count matches do not count as coupling evidence. This branch is opt-in and has not been registered as a new formal policy or evaluated for real predictive gain.

## 4. Experiments To Complete

### 4.1 Completed Development Evidence

The frozen v1 protocol fits ETH, Hotel and Zara01/02/03, with all Zara recordings
kept in one producer fold. Development uses Students01/03, one University site.
These historically exposed data are explicitly not independent confirmation.
The task uses native annotation steps, not verified metric/time calibration.

| Seed | Neural normalized ADE gain vs CV (%) | Candidate/floor oracle gain (%) | Selected policy |
| --- | ---: | ---: | --- |
| 17 | -7.093 | 0.722 | CV floor |
| 29 | -7.900 | 0.704 | CV floor |
| 43 | -7.249 | 0.441 | CV floor |

The oracle uses future labels for diagnosis only. There are 14,920 complete
development paths, 14,931 valid endpoints and 30,438 past-supported queries;
these are distinct denominators. Fifteen neural fits complete 1,000 updates each.
One development site cannot provide a meaningful site-bootstrap interval;
seed variability is not new-site uncertainty. Easy baseline error is near zero,
so absolute excess accompanies relative degradation. No successful intervention,
coupling gain or safety guarantee follows from selecting the unchanged floor.

Fit-only audits show approximately 1% of ETH and Zara02 windows account for
96.32% and 99.74% of squared normalized target energy. Development normalized
error is also highly concentrated: the top approximately 1% of Students03 rows
accounts for 62.49% of CV normalized error but 1.09% of native-coordinate error.
This motivates a separately versioned one-factor Smooth-L1 loss ablation, not
post hoc replacement of the primary metric. Details and all failed controls are
in [the result package](8to12_development_v1/conclusions.md).

### 4.2 Remaining Mechanism and Confirmation Tests

The primary mechanism test holds candidate forecasts and training examples fixed while varying cost supervision and joint selection. At matched actual intervention counts, improved forecast composition would support a narrower contribution than a new predictor architecture. A gain that disappears after matching counts, or a lower proximity penalty accompanied by worse forecasting, would not support that claim. Real accuracy, independent-scene risk calibration and physical safety remain separate questions; none is established by the analytical examples in the assumption audit.

The [deferral control](deferral_control/method_and_limits.md) fits linear or small neural routing on the same causal rollout features and held-fold predictions as the relative-cost head. It preserves continuous error weights and makes its bounded-cost transform explicit in the protocol. Clipped training risk, unclipped forecasting error and calibrated safety are distinct quantities. Its [development comparison](deferral_development/implementation_and_limits.md) now verifies identical OOF training inputs and runs all controls on identical forecasts before labels are read. The unconstrained deferral arm is not claimed to share M3W's budget or coverage. Paired scene-level error differences are descriptive; synthetic integration and recovery do not establish a real accuracy advantage. The real protocol and frozen confirmation family have not been changed.

An [EqMotion-core adapter](public_baselines/compatibility_and_limits.md) now provides an externally sourced predictor for this implementation path. The source is pinned and checked before loading; past-only context replaces the release's future-availability-dependent preprocessing. The current adapter trains one fixed output head with the local MSE objective. It is an adapted K=1 control, not a reproduction of published best-of-20 results or evidence of comparative accuracy. Synthetic CPU/MPS training and real-input integrity checks are complete; real fitting and matched-context, matched-budget comparisons are not. AgentFormer's author-reported normalization correction must also be respected when constructing the eventual public-baseline table.

Compare the same predictor with no gate, independent confidence gating, expected-error gating, scene-uniform gating and joint intervention selection. Include simple regressors and modern published predictors so that benefits cannot be attributed only to replacing a weak baseline. Separate K=1 from best-of-K evaluation. Deduplicate underlying recordings across dataset distributions, freeze development and calibration choices, and use a final confirmation set with no prior model-selection exposure. Report per-domain raw ADE/FDE, normalized supplemental errors, easy/hard slices, joint errors, proximity proxies, coverage, latency, three seeds and scene-level confidence intervals.

The [frozen final-family evaluator](confirmation_evaluation/implementation_and_limits.md) now checks actual predictor and out-of-fold producer seeds, consumes completed calibration decisions, and reports all prespecified controls without final-set model selection. Seed-mean errors are distinguished from ensemble predictions; positive harm is computed within each seed before averaging, and easy-case preservation remains visible per seed. Paired bootstrap draws share physical-scene blocks across arms and seeds, conditional on the fixed models. They are descriptive, not multiplicity-adjusted guarantees. Synthetic three-seed training and final-evaluation recovery have been exercised; the real comparison remains unrun under the still-unapproved protocol.

No confirmatory results are available for the proposed method. The historical Stage44 no-scene result (+37.49% all, +20.32% t50 normalized four-waypoint ADE vs an interpolated floor) is motivation only and must not appear as a final main-table result without a corrected independent evaluation.

## 5. Limitations

The current datasets are represented in pixel or dataset-local coordinates with unverified cross-source scale and effective time. Physical collision risk cannot be inferred from a normalized proximity threshold alone. Scene proxies are not verified semantic maps. Most historical observations overlap in time, and some named dataset collections may contain the same underlying recordings. Calibration assumptions may fail under arbitrary shift, and a small number of independent scenes may make safety bounds uninformative. A selective predictor is not an action-conditioned simulator or a foundation world model.

The new [support audit](risk_calibration/support_audit.md) makes this limitation concrete: nine current canonical recordings correspond to six physical-scene groups, all historically development-exposed. With six hypothetical independent calibration scenes and zero observed [0,1] loss, even one policy/one risk gives an upper bound of 0.4996 at illustrative delta=0.05 under the current Hoeffding screen. This is a sensitivity calculation, not an actual calibration or universal sample-complexity lower bound. Repeated windows cannot improve independent-scene support. Until data roles and independent confirmation are resolved, useful formal risk control must remain an unestablished part of the proposed contribution rather than a result.

A separate [CITR diagnostic conversion](citr_causal_intake/implementation_and_limits.md) now retains synchronized pedestrian and vehicle raw positions with row-level provenance. Its 38 controlled clips cover only one physical site, so the conversion cannot be counted as 38 independent calibration scenes. No data-use role, independent-test eligibility, typed neural training or predictive result has been established for this cache. It may support future controlled mechanism checks after review, not a current external-generalization claim.

The subsequent [DUT diagnostic intake](dut_causal_intake/implementation_and_limits.md) adds 28 natural-campus clips at two source-described locations, not 28 independent scenes. Exact source-row checks identified two simultaneous pedestrian IDs with identical 145-frame trajectories in one clip; that recording is flagged for quality quarantine before formal use. Raw-coordinate semantics also require care because the author preprocessing divides raw positions by a scale despite a general meter statement in the README. Source terms, scientific roles, annotation resolution and prior-use eligibility remain open. This acquisition supplies neither a confirmatory performance result nor enough independent scenes to establish the proposed risk guarantee.

A separate [admission check](intake_admission/implementation_and_limits.md) now prevents a scientific-role declaration from clearing pending source review or a bound annotation quarantine. It follows the source, conversion and quality evidence and checks role-specific review declarations before opening new-source data through the experiment contract. These engineering controls do not authenticate permission or establish independent sampling, and they are not offered as a methodological contribution or evidence of forecasting improvement.

## 6. Reproducibility

The completed v1 package stores canonical recording IDs, hash-bound protocol,
past-only schemas, train-only preprocessing, full held-fold producer lineage,
fixed development policies, seeds, atomic checkpoints and heartbeat logs.
Commit `707d4017` preserves its exact training implementation. Fresh real fitting
is established, but independent calibration and confirmation are not. Source
identity is checked on resume; old model hashes must not be edited to bypass a
newer implementation mismatch. Subsequent ablations use new protocol versions.

## Verified References

- AgentFormer: https://ye-yuan.com/agentformer/
- EqMotion: https://openaccess.thecvf.com/content/CVPR2023/html/Xu_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_CVPR_2023_paper.html
- SingularTrajectory: https://arxiv.org/abs/2403.18452
- Joint Metrics Matter: https://arxiv.org/abs/2305.06292
- Conformal Risk Control: https://arxiv.org/abs/2208.02814
- Learn then Test: https://arxiv.org/abs/2110.01052
- SODA-MPC: https://proceedings.mlr.press/v283/contreras25a.html
- Generalized HCP: https://arxiv.org/abs/2608.15500
- HCP, inspected preprint v4: https://arxiv.org/abs/2306.06342v4
- CAFHT, ICML 2024: https://proceedings.mlr.press/v235/zhou24l.html
- Adaptive CP for motion planning, L4DC 2023: https://proceedings.mlr.press/v211/dixit23a.html
- Consistent Estimators for Learning to Defer to an Expert: https://proceedings.mlr.press/v119/mozannar20b.html
- Regression with Multi-Expert Deferral: https://proceedings.mlr.press/v235/mao24d.html
- Selective Regression under Fairness Criteria: https://proceedings.mlr.press/v162/shah22a.html
