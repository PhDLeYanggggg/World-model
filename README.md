# M3W: Real-World Multimodal Agent-Scene World Model

M3W is my research project on top-down multi-agent world modeling. I study
when a neural forecast adds value over a strong motion baseline, and how to
avoid harming the cases that the baseline already handles well.

The long-term aim is a useful agent-scene world model. The current work is
narrower: causal forecasting, learning the gain and harm of an intervention,
and deciding which agents should receive a neural prediction.

## Research Question

The main observation protocol is **8 observed steps and 12 predicted steps**.
The current European-source experiment uses a stride of 12 raw frames;
raw-frame `t+50` is a separate historical supplement, not the same task.

I am testing three linked questions:

1. Can the neural predictor contribute something beyond a strong causal baseline?
2. Can past-only context identify where switching helps and where it hurts?
3. Can joint decisions improve on independent selection at the same intervention count?

JEPA, Transformer and hybrid models are part of the research track. Combining
these modules is not, by itself, evidence of a useful world model or a new method.

## Latest Experiment

**Temporal supervision improves prediction error, but not safe decisions.**
I completed all 216 matched neural cost-head fits on the frozen EuropeanSquares
source partitions, with three head seeds and 2,000 updates per fit. The comparison
keeps the architecture, primary loss and sampling fixed, and varies no auxiliary,
row-mean auxiliary and temporal auxiliary supervision. All models were frozen
before the seven-arm development evaluation.

Temporal supervision improves normalized signed-score error over the row-mean
control by 0.024422, with a nominal locality interval of [0.001405, 0.054556].
But its paired decision utility is worse, including when intervention counts
are matched within each query. It also loses to the no-auxiliary neural control.
Sixty of 72 source/seed views already violate the 2% selected easy-harm budget
on known outcomes. This cannot be explained only by missing labels.

The experiment ran successfully; the proposed model did not pass. I am retaining
the negative result and investigating false-safe harm/reference predictions,
without relaxing the risk budget or tuning the completed evaluation. These are
exposed development findings, not independent confirmation or a deployment gain.

The follow-up replay now shows that the failure starts on the training data.
All 216 heads underestimate easy-case harm on their selected populations.
The temporal arm violates the fixed budget in 52 of 72 TRAIN source/seed views;
its median predicted/actual selected-harm ratio is only 0.0392. Reference-cost
overestimation is not the dominant error. This is a training-set diagnosis, not
a new generalization gain. I am testing a more suitable easy-harm loss next.

The paired easy-harm experiment is now registered: 144 fixed-budget fits, with
the original quadratic loss as an exact control. The real TRAIN pilot is complete:
checkpoint recovery and the original control replay exactly, and the resource
checks pass. The two 100-step pilot monitoring losses did not improve, so this
is execution evidence, not an accuracy or safety result. The full comparison
is now submitted as four disjoint CREATE tasks, with 2,000 updates per fit.
The first full task reached2,000 updates but failed exact equality with its
historical control; the reported weight difference is2.43e-6. Unstarted tasks
are held while a same-node TRAIN replay diagnoses the mismatch. It is not yet
known whether the low-level cause is a historical floating-point/kernel path.
Another shard completed36 fits; all checkpoints and its18 historical controls
verify exactly. Those valid results are retained, not retrained.
The completed same-node diagnostic now reproduces the failed checkpoint exactly
with the unmodified original trainer, both directly and through pilot recovery.
Only the historical saved state differs. An explicit, pre-readout amendment
uses that exact original-trainer replay for this one identity and keeps the
historical mismatch visible; floating tolerances and research gates do not change.
All final heads must be verified before the fixed development evaluation.
The three unfinished shards have been resubmitted; the last scheduler check
shows them queued. The cache-free six-arm readout is implemented and registered,
with61 scoped tests passing, but has not run on real development data yet.
[Pilot evidence and limits](outputs/publication_readiness_2026_09/european_easy_harm_deviance_v1/pilot_report.md).

**Execution update, 6 October at 12:56 UTC:** two resumed shards have completed.
There are now 110 hash-verified final fits, including 54 exact historical controls
and the one explicitly documented original-trainer replay. The last shard failed
on another identity, with a reported weight difference of 0.113232. This is not
dismissed as a negligible numerical discrepancy. A TRAIN-only original/new
implementation sweep is registered for the 17 still-unaccepted pairs. Existing
fits are preserved, no additional acceptance exception has been granted, and
new development evaluation remains closed.
The diagnostic has since completed: all 17 new/original implementation pairs
match exactly, but none matches its historical shard-zero checkpoint. The
historical compute node no longer exists. This localizes an execution-reproduction
problem without proving its low-level cause; I am retaining it explicitly while
preparing the next recovery contract. It is not a scientific result for the loss.
[Diagnostic evidence and limits](outputs/publication_readiness_2026_09/european_easy_harm_deviance_v1/control_diagnostic_v2/conclusions.md).

The [version-three English manuscript](outputs/publication_readiness_2026_09/evidence_manuscript_v3/manuscript.md)
connects the earlier SDD study to these European development results without
pooling their metrics. It retains strong conventional controls and negative
findings; the [new experimental addendum](outputs/publication_readiness_2026_09/european_temporal_auxiliary_v1/readout/manuscript_addendum.md)
replaces its earlier pending-training status. A second
[TRAIN selection diagnostic](outputs/publication_readiness_2026_09/evidence_manuscript_v3/train_selection_diagnostic_addendum.md)
shows that selected harm is already underestimated before transfer, with a
traceable decomposition and the successor's reproducibility limitation retained.
The central question remains whether better cost prediction produces
better decisions on the selected population, not merely a smaller average loss.

- [Completed temporal diagnostic](outputs/publication_readiness_2026_09/european_temporal_target_audit_v1/conclusions.md)
- [Complete matched training and downstream results](outputs/publication_readiness_2026_09/european_temporal_auxiliary_v1/readout/report.md)
- [Why the neural policy failed](outputs/publication_readiness_2026_09/european_temporal_auxiliary_v1/readout/failure_analysis.md)
- [TRAIN false-safe diagnosis and next hypothesis](outputs/publication_readiness_2026_09/european_temporal_auxiliary_v1/false_safe_train_diagnostic_v1/conclusions.md)
- [Registered training and recovery](outputs/publication_readiness_2026_09/european_temporal_auxiliary_v1/reproduction_zh.md)
- [Current paper and reproducible tables](outputs/publication_readiness_2026_09/evidence_manuscript_v3/README.md)
- [Complete results ledger](README_RESULTS.md)

### Earlier Conditional Harm Result

**A positive harm head repairs part of the failure, but still does not pass.**
I trained all72 fixed conditional harm heads, retaining the original tree routes
and raw benefit/reference estimates. The new output removes negative-to-zero
harm clipping, while a training-only normalizer preserves each leaf's mean.

Compared with the same-feature additive control, known-label risk violations
fall from20 to2. Complete risk support rises from33 to37 relative to the original
model. However, risk upper violations increase from7 to11, and normalized
signed-score prediction error worsens by0.1362, nominal95% interval[0.0398,0.2576].
Nine remaining upper violations involve unknown outcomes; two already fail on
known labels. Neither better support elsewhere nor lower training loss excuses
these failures. I am not advancing this model to transfer or deployment.

The full native-arm64 run took22minutes, with exact refits and inference replay.
All72 checkpoints are verified in owned CREATE storage;37 scoped tests pass.
This is a combined output-form/loss experiment on exposed development scenes,
not independent confirmation or new neural trajectory dynamics.
[Results and limitations](outputs/publication_readiness_2026_09/european_positive_harm_v1/conclusions.md),
[reproduction](outputs/publication_readiness_2026_09/european_positive_harm_v1/README.md).

### Preceding Component Diagnosis

**The new information helps, but the harm predictions allow unsafe expansion.**
I completed all 13 fixed component interventions on 72 frozen cost heads.
Nearly all newly accepted actions had previously failed the easy-risk condition.
Their known-label selected easy-harm ratio averages 7.13% across the development
localities, above the unchanged 2% budget. Harm underestimation is the main
average contributor; benefit and reference corrections are not.

Removing the total-harm correction reduces failures, but still does not restore
the original risk support. It is a diagnostic, not a deployable shortcut. It
motivated the conditional harm experiment above, preserving useful past features
and original benefit/reference estimates without loosening thresholds.
All frozen inference replays exactly; 14 scoped tests pass. No new model was
trained in this diagnosis and no independent evaluation was opened.
[Findings and limitations](outputs/publication_readiness_2026_09/european_quality_components_v1/conclusions.md).

### Preceding Training Result

**Past observation quality helps prediction, but does not yet make selection
safe.** I completed all 144 paired auxiliary fits on the 72 frozen cost forests.
The actual past-quality features outperform a within-recording shuffled control,
with no future-quality input or change to the risk budget.

Validation cost error decreases, and conservative utility improves even when
each recording/frame has the same intervention count: +0.0618% of full known
reference mass versus the original model, nominal interval [0.0204%, 0.1074%].
This is a decision-utility contrast, not an ADE/FDE improvement. However, complete
risk support falls from 33 to 21 groups, and selected easy-risk upper violations
increase from seven to 41. Twenty already violate the budget using known labels.
The information signal is useful; this decision rule is not deployable.

All fits, serialized predictions and readouts reproduce exactly. Checkpoints
are verified in owned CREATE storage; 19 scoped tests pass. The component
diagnosis above now localizes the unsafe expansion, while
keeping thresholds and independent evaluation roles unchanged.

**Incomplete labels are not the main explanation for the remaining mistakes.**
After a relative-target refit failed to improve safety, I traced the frozen
models' decisions back to the original recordings. All 318,969 source rows and
their available future coordinates match the tracker exports. This rules out a
detected packing error, but does not establish that the tracker labels are correct.

Most observed harm occurs even when all 12 future labels are present: 73.1% after
equal weighting across the localities with defined harm, with a nominal 95%
interval of [61.1%, 84.2%]. Four complete-label groups still exceed the unchanged
2% selected-harm budget. Dropping incomplete trajectories would not solve this.

The completed diagnostic reused frozen models, replayed the full analysis, and
passed 19 scoped tests. It motivated the paired training experiment above;
neither experiment filters cases using future label quality.

These are exploratory development results, not independent confirmation or a
deployment upgrade. The detailed ledger retains the negative calibration,
neighbor-history and relative-target experiments that motivated this check.

- [Conditional harm findings](outputs/publication_readiness_2026_09/european_positive_harm_v1/conclusions.md)
- [Failure analysis](outputs/publication_readiness_2026_09/european_positive_harm_v1/failure_analysis.md)
- [Reproduction and evidence status](outputs/publication_readiness_2026_09/european_positive_harm_v1/README.md)
- [Preceding frozen-component diagnosis](outputs/publication_readiness_2026_09/european_quality_components_v1/conclusions.md)
- [Preceding paired training result](outputs/publication_readiness_2026_09/european_past_quality_auxiliary_v1/conclusions.md)
- [Preceding raw-label diagnostic](outputs/publication_readiness_2026_09/european_label_support_v1/conclusions.md)
- [Complete results ledger](README_RESULTS.md)

## Fixed-Upstream Seed Control

**The remaining risk failures are not just an unlucky random seed.** I fixed
the upstream predictors and trained 48 additional cost heads with different
seeds, keeping data, features, targets and model settings identical. The 24
original heads remain in the comparison; no seed was selected after evaluation.

The three heads still violate the 2% selected easy-harm budget in 7/30, 5/30 and
6/30 defined directions. Each has another 42 undefined directions, not passes.
Same-count utility differences are effectively zero, with intervals crossing
zero. The small average ADE gains do not make these policies safe to deploy.

The diagnostic distinguishes underestimated harm from overestimated reference
error. The latter can make a harmful switch appear safe even when harm itself
is overpredicted. This motivated the source-only component calibration experiment
above, rather than another seed or threshold search.

The first full fit and entire readout replay exactly; 34 scoped tests pass.
These remain exposed-development results. Independent confirmation is closed,
the deployment floor is unchanged, and there is no submission-ready claim.

- [Crossed-seed results](outputs/publication_readiness_2026_09/european_crossed_head_seed_v1/results.md)
- [Six-case failure analysis](outputs/publication_readiness_2026_09/european_crossed_head_seed_v1/failure_analysis.md)
- [Design and reproduction](outputs/publication_readiness_2026_09/european_crossed_head_seed_v1/operation.md)

## Nonlinear Cost Control

**A standard nonlinear cost model improves preservation, but is not a safe
upgrade.** I trained72 fixed ExtraTrees models on the same European source
partitions and causal information as the neural cost head. Source-validation
cost error is not clearly better. At the same per-query intervention count,
ADE is worse by0.149% (nominal locality interval[-0.252%,-0.061%]).

Source screening limits worst whole-easy degradation to0.032%, but selected easy
harm still exceeds the2% budget in6/68 defined directions, with a worst ratio of
7.55%.148 directions remain undefined. All six failures involve seed43, which
also affects upstream models; it would be invalid to discard that seed.

All216 development readouts replay exactly;30 scoped tests pass. These are
exposed-development results, not independent confirmation. I am keeping the
deployment floor unchanged and isolating cost-head versus upstream seed effects
next, rather than claiming that another estimator solved the problem.

- [Results and comparisons](outputs/publication_readiness_2026_09/european_source_forest_v1/results.md)
- [Failure analysis](outputs/publication_readiness_2026_09/european_source_forest_v1/failure_analysis.md)
- [Reproduction](outputs/publication_readiness_2026_09/european_source_forest_v1/operation.md)

## Previous Control

**Handling missing labels restores a little coverage, not a learned or safe
upgrade.** I replaced the blanket missing-outcome veto with a conservative
geometric bound, keeping the same candidates and 2% selected-harm budget.
The frozen follow-up improves ADE by 0.03050% over the previous policy
(nominal locality interval [0.01070%, 0.05157%]), but the same-count comparison
is exactly zero. Every retained source head is still a step-zero prior.

Intervention rises from 0.264% to 2.847%. Overall easy ADE stays preserved, yet
5/27 nonempty views violate easy selected positive-harm risk, with a worst ratio
of 4.54%. The other 189 views are undefined, not safety passes. This is a small
coverage gain on exposed development data, not a deployment or neural-model
improvement. All216 readouts replay exactly and 27 scoped tests pass.

- [Policy contrast results](outputs/publication_readiness_2026_09/european_completion_screen_policy_v1/results.md)
- [Failure analysis and next control](outputs/publication_readiness_2026_09/european_completion_screen_policy_v1/failure_analysis.md)
- [Reproduction](outputs/publication_readiness_2026_09/european_completion_screen_policy_v1/operation.md)

## Missing-Outcome Support

**Missing-outcome bounds recover some source support, not learned superiority.**
Of 41 frozen candidates rejected only for missing selected labels, nine pass
a conservative maximum-disagreement bound under the original 2% risk budget.
The other 32 still lack easy-risk support. All 149 candidates already failing
known-outcome risk remain unsupported.

The nine recovered views include only three trained-checkpoint views, and their
initial priors have higher known validation utility. All 216 bound readouts
replay exactly; 1,512 independent arithmetic checks and 20 scoped tests pass.
No new policy, transfer gain or deployment success follows from this diagnostic.

- [Missing-outcome results and limitations](outputs/publication_readiness_2026_09/european_unknown_outcome_bounds_v1/results.md)
- [Bound proof and frozen protocol](outputs/publication_readiness_2026_09/european_unknown_outcome_bounds_v1/protocol.md)
- [Reproduction](outputs/publication_readiness_2026_09/european_unknown_outcome_bounds_v1/operation.md)

The follow-up contrast above is complete: 63 fallback and nine step-zero prior
choices, with no trained head selected. Source support does not by itself
establish transferred conditional-risk control.

## Decision-Aware Control

**Decision-aware source validation mostly abstains; it does not repair learned
selection.** I tested a fixed utility-and-risk rule over three frozen heads per
source, keeping the original 2% selected-reference budget. Of 72 source choices,
69 fall back. The other three retain a step-zero prior, not a trained improvement.

The resulting policy intervenes on only 0.264% of rows on average, with +0.00382%
ADE gain over the floor. Same-count actions are identical to its comparator, so
the primary gain is exactly zero. Whole-population easy ADE is preserved, but
two of the nine defined easy-risk views still violate the budget; 207 views are
undefined and 53 selected outcome occurrences are unknown. **No deployment
upgrade or neural-contribution claim.**

All source choices and all 216 transfer readouts replay exactly, with 27,000
independent metric checks and 14 scoped tests. Independent calibration and
confirmation remain closed. The next question is whether causal disagreement
bounds can resolve any missing-outcome support without ignoring unknown harm;
known-outcome conditional-risk failures remain a separate problem.

- [Decision-aware control results](outputs/publication_readiness_2026_09/european_source_policy_selection_v1/results.md)
- [Failure analysis and evidence gap](outputs/publication_readiness_2026_09/european_source_policy_selection_v1/failure_analysis.md)
- [Reproduction](outputs/publication_readiness_2026_09/european_source_policy_selection_v1/operation.md)

## Checkpoint Control

**Better global cost prediction did not produce better intervention decisions.**
I completed a source-internal checkpoint-selection control: 72 Torch fits,
three seeds, and 216 frozen directional evaluations across 12 development
localities. Checkpoints were chosen on disjoint recordings from their training
locality, never on the transfer locality.

| Validation-selected versus final checkpoint | Result | Nominal 95% locality interval |
|---|---:|---:|
| Signed-score MSE difference, lower is better | -0.70681 | [-0.97098, -0.45677] |
| Same-count ADE improvement | -0.53953% | [-0.73862%, -0.34152%] |

The selected head avoids 0.29381 percentage points of harm but loses 0.81194
points of benefit on the full-floor diagnostic denominator. It still violates
the original easy selected-harm screen in 148/216 views, with 21 undefined;
worst-view easy ADE degradation is 21.54%. **No deployment upgrade.** This result
separates a useful model-selection improvement from an unsuccessful policy.

The first full fit and complete readout replay exactly. Independent arithmetic
checks 27,000 metric values and 747,900 per-query intervention counts; 27 scoped
tests pass. These are exposed development results, not independent confirmation.
Independent selection, calibration and confirmation remain closed.

- [Results](outputs/publication_readiness_2026_09/european_source_checkpoint_v1/results.md)
- [Failure analysis](outputs/publication_readiness_2026_09/european_source_checkpoint_v1/failure_analysis.md)
- [Reproduction](outputs/publication_readiness_2026_09/european_source_checkpoint_v1/operation.md)

## Preceding Risk Diagnosis

The frozen-model diagnosis is complete locally. **Selected easy harm is already
underestimated on training sources, and transfer makes it worse.** At matched
intervention counts, the nonlinear head predicts 0.32% easy positive-harm risk
on fitting sources versus 7.90% observed; on internal transfer, it predicts 0.31%
versus 14.01% observed. These selected-harm ratios are not overall ADE degradation.

All288 input packets match the committed hashes and replay exactly. CREATE
job37602475 is now collected and verified after access recovered: all numeric
summary fields agree with the local run within strict roundoff tolerance. Its
remote replay is exact; the two files retain distinct execution-provenance text.
No new local row cache or model training. Numerical replication does not turn
the failed-risk result into a deployment upgrade.

- [Risk decomposition](outputs/publication_readiness_2026_09/european_boundary_diagnostic_v1/local_results.md)
- [Failure analysis](outputs/publication_readiness_2026_09/european_boundary_diagnostic_v1/failure_analysis.md)
- [Execution and evidence boundaries](outputs/publication_readiness_2026_09/european_boundary_diagnostic_v1/local_operation.md)
- [CREATE recovery and complete-summary verification](outputs/publication_readiness_2026_09/european_boundary_diagnostic_v1/create_recovery_20261001.md)

## Previous Capacity Control

I tested whether a more expressive cost head can transfer useful gain and harm
estimates between development localities. Each head trains on one locality and
is evaluated on another, with preprocessing confined to the training source.
Both arms use identical causal inputs, losses and sampled queries. The comparison
completed 144 heads and 288,000 updates, then froze decisions before readout.

The result is mixed, and **the new head is not promoted**:

| Comparison | Result | Nominal 95% locality interval |
|---|---:|---:|
| Nonlinear minus affine signed-score MSE, lower is better | +0.07709 | [-0.10209, +0.30087] |
| Same-count nonlinear versus affine ADE improvement | +0.65629% | [+0.34267%, +1.00558%] |

Better average selection does not establish reliable risk prediction. At the
same intervention count, selected all-risk violations increase from 158 to 168
of 216 dependent views. The nonlinear head's worst easy-case degradation is
4.18%, beyond the 2% limit. The primary prediction-error comparison does not
establish an improvement, despite lower training loss.

These are internal cross-fitted results on 12 already exposed localities, not
independent confirmation or a new neural dynamics forecast. Independent
selection, calibration and confirmation sources remain closed, and deployment
is unchanged. I report unknown-label interventions rather than counting them
as harmless.

- [Results and evidence boundaries](outputs/publication_readiness_2026_09/european_inner_separability_v1/results.md)
- [Failure analysis and next controlled question](outputs/publication_readiness_2026_09/european_inner_separability_v1/failure_analysis.md)
- [Training protocol](outputs/publication_readiness_2026_09/european_inner_separability_v1/protocol.md)
- [Earlier fitting-switch diagnosis](outputs/publication_readiness_2026_09/european_fitting_switch_diagnostic_v1/results.md)

## Previous Model Comparison

**Freezing easy-occurrence probabilities did not improve neural selection.**

I tested whether keeping an existing occurrence estimate fixed would protect it
while learning conditional costs. Both arms used identical split networks, warm
starts and training queries. The experiment trained 216 heads for 2,000 updates
each, then froze every decision before reading development outcomes.

| Same-count comparison | ADE improvement | Nominal 95% locality interval |
|---|---:|---:|
| Fixed versus trainable occurrence | -0.01639% | [-0.02903%, -0.00597%] |
| Fixed versus original raw-risk control | -0.05500% | [-0.10730%, -0.00443%] |

The fixed head had slightly better held-development probability and risk-error
point estimates, but selected worse predictions. It avoided 0.00529 percentage
points of positive harm while losing 0.02146 points of benefit on the common
full-floor diagnostic denominator. That tradeoff does not repair selected risk:
89 of 216 dependent views still violate the risk screen, and 22 remain undefined.
Easy preservation passes the developmental check. **No deployment upgrade.**

All 108 action groups and the complete readout replay exactly; 68 scoped tests
pass. A null optimizer certificate was handled through the existing checked
fallback, preserving the original 95 completed groups and sealed algorithms.
This repairs an engineering interruption, not the optimizer's underlying numerical
failure or the model's risk estimates. Independent confirmation remains closed.

This result narrows the next question: how to retain beneficial interventions
without admitting false-safe ones. Better average probability fit alone is not
enough. These remain developmental risk-controller experiments, not a new world
dynamics result, a safety certificate or a submission-ready system.

- [Result, interpretation and next question](outputs/publication_readiness_2026_09/european_fixed_occurrence_v1/conclusions.md)
- [All controls, metrics and gates](outputs/publication_readiness_2026_09/european_fixed_occurrence_v1/results.md)
- [Benefit/harm accounting](outputs/publication_readiness_2026_09/european_fixed_occurrence_v1/failure_analysis.md)
- [Verification scope](outputs/publication_readiness_2026_09/european_fixed_occurrence_v1/verification_report.md)
- [Chinese run and recovery guide](outputs/publication_readiness_2026_09/european_fixed_occurrence_v1/operation_zh.md)

## Earlier Risk-Priority Result

**A small accuracy recovery did not repair selected risk.**

I tested whether limiting auxiliary gradients could improve risk-constrained
neural selection. The paired experiment trained 216 risk heads for 2,000 updates
each, keeping forecasts, architecture, initialization and sampled queries fixed.
Only the auxiliary-gradient norm cap changed. Every action was frozen before
the development readout.

| Same-count risk-priority comparison | ADE improvement | Nominal 95% locality interval |
|---|---:|---:|
| Versus uncapped auxiliary training | +0.00375% | [+0.00129%, +0.00681%] |
| Versus the original raw-risk control | -0.01435% | [-0.02330%, -0.00611%] |

The repair retained slightly more useful switches, but also increased positive
harm. Selected-risk violations rose from 51 to 54 of 216 dependent views; both
matched policies had 25 views with undefined selected risk. Easy preservation
passed its developmental screen, but the risk screen did not. The original
raw-risk control remains more accurate at matched counts. **No deployment upgrade.**

This followed an earlier negative result: explicit easy-occurrence supervision
improved probability estimates but worsened selection. A fitting-only diagnostic
then measured auxiliary-gradient dominance. The new experiment tests one fixed
response to that imbalance; it does not establish that optimization imbalance
explains the remaining generalization or calibration failures.

All 108 action groups and the complete readout replay exactly. Independent
arithmetic checks and 47 scoped tests pass. All uncapped training controls
reproduce the previous states; the first repaired pair was retrained for exact
replay. These checks support reproducibility, not efficacy. The experiment uses
twelve already-opened development localities, not independent confirmation.
Undefined risk, adverse slices and negative comparisons remain in the results.

I also checked why the repair underperformed on its own fitting data. Full
factor accounting shows a2.11% increase in the risk objective: occurrence
estimation worsened, while changes in the cost factors partly cancelled its
error. Better combined scores therefore need not mean better component estimates.
All108 diagnostic groups replay exactly. This narrows the next controlled test;
it does not establish a generalization mechanism or overturn the failed risk gate.

- [Conditional-risk diagnosis](outputs/publication_readiness_2026_09/european_easy_component_diagnostic_v1/conclusions.md)
- [Restored fitting-only benefit labels](outputs/publication_readiness_2026_09/european_fitting_gain_labels_v1/results.md)

- [Results and every control](outputs/publication_readiness_2026_09/european_easy_risk_priority_v1/results.md)
- [Decision, interpretation and limits](outputs/publication_readiness_2026_09/european_easy_risk_priority_v1/conclusions.md)
- [Benefit/harm accounting](outputs/publication_readiness_2026_09/european_easy_risk_priority_v1/failure_analysis.md)
- [Locality, seed and tail results](outputs/publication_readiness_2026_09/european_easy_risk_priority_v1/locality_seed_quality.md)
- [Training evidence](outputs/publication_readiness_2026_09/european_easy_risk_priority_v1/training_report.md)
- [Verification scope](outputs/publication_readiness_2026_09/european_easy_risk_priority_v1/verification_report.md)
- [Chinese reproduction guide](outputs/publication_readiness_2026_09/european_easy_risk_priority_v1/operation_zh.md)
- [Prior supervision experiment](outputs/publication_readiness_2026_09/european_easy_hurdle_v1/conclusions.md)
- [Gradient diagnostic](outputs/publication_readiness_2026_09/european_easy_gradient_diagnostic_v1/results.md)

## Previous Controlled Result

**Correcting average risk bias did not produce a better deployment policy.**

After training 216 risk heads, I tested whether their remaining fitting-only
score bias explained unsafe switching. The forecasts and learned utilities
were frozen; only two already-fitted risk offsets changed. All 108 action groups
were committed before the new development readout.

The offsets greatly reduced intervention, but also removed useful predictions.
At exactly the same intervention count in each query, centered risk was worse
than the original-risk control:

| Frozen risk head | ADE gain versus same-count control | Nominal 95% locality interval |
|---|---:|---:|
| Pointwise | -0.00335% | [-0.00882%, -0.00017%] |
| Subset aggregate | -0.00523% | [-0.01182%, -0.00043%] |

For the aggregate head, intervention fell from 7.80% to 0.67%. Two hundred of 216
dependent views then entirely abstained; six remaining views still violated
the 2% selected-harm screen. Easy cases were preserved, but this is **not a safe
deployment upgrade**. Lower fitting loss and fewer harmful switches are not
enough when useful switches disappear too.

All 108 action groups and their evaluation replay exactly. **34 scoped tests
pass**; separate arithmetic checks cover 1,495,800 query/head constraints,
3,456 cost views and 23,714 locality reductions. These repeated contexts are
not independent samples. A replay-only JSON identity-format defect was fixed
without changing original code or scientific outputs.

Most rejected original admissions triggered the easy-risk constraint. This
motivated the paired easy-occurrence experiment above. It improved probability
and conditional-cost fit, but did not repair allocation or selected risk.
Independent selection, calibration and confirmation remain closed.

- [Centered-risk experiment and failure analysis](outputs/publication_readiness_2026_09/european_centered_risk_policy_v1/conclusions.md)
- [New paired repair: audit, execution status and remaining work](outputs/publication_readiness_2026_09/european_easy_hurdle_v1/results.md)
- [Completed training and exact replay evidence](outputs/publication_readiness_2026_09/european_easy_hurdle_v1/training_result.md)
- [All controls, intervals and risk failures](outputs/publication_readiness_2026_09/european_centered_risk_policy_v1/results.md)
- [Frozen verification record](outputs/publication_readiness_2026_09/european_centered_risk_policy_v1/verification.json)
- [Current reproduction guide](outputs/publication_readiness_2026_09/european_centered_risk_policy_v1/operation_zh.md)

- [What the frozen decisions reveal](outputs/publication_readiness_2026_09/european_selected_risk_diagnosis_v1/conclusions.md)
- [Residual and support results](outputs/publication_readiness_2026_09/european_selected_risk_diagnosis_v1/results.md)
- [Diagnostic verification](outputs/publication_readiness_2026_09/european_selected_risk_diagnosis_v1/verification.json)
- [Fitting-only bias probe](outputs/publication_readiness_2026_09/european_signed_bias_probe_v1/conclusions.md)
- [Bias-fit results and verification](outputs/publication_readiness_2026_09/european_signed_bias_probe_v1/results.md)
- [Parent neural experiment](outputs/publication_readiness_2026_09/european_subset_excess_v1/results.md)
- [Chinese reproduction guide](outputs/publication_readiness_2026_09/european_selected_risk_diagnosis_v1/operation_zh.md)

The earlier fixed-predictor allocation experiment did improve ADE at matched
counts, but failed observed-risk control. The subsequent pure query-aggregate
loss also failed. I retain these negative results because they distinguish
better allocation, better prediction loss and reliable risk control.

## Evidence Boundaries

The project currently remains a **2.5D multi-agent trajectory/world-state
research system**, not a true 3D or foundation world model. Coordinates are
image-local, pixel-space or dataset-local unless separately verified. Raw-frame
horizons are not seconds. Detector, inferred and self-audited labels are not
human gold, and statistical harm diagnostics are not physical-safety guarantees.

Future targets are used for loss and evaluation, never as inference features.
Central velocity and test-endpoint goals are excluded. Source exclusion applies
to upstream predictors, preprocessing, teachers and downstream heads. Supplied
historical annotations can still have interpolation limitations: past-indexed
access alone does not establish strict sensor-as-of availability.

Earlier external Stage35/37/43/44 scores have recording, teacher-exposure or
selection limitations. They remain exploratory; renaming their splits cannot
make them independent evidence. Current percentages are not directly
comparable with those historical `t+50` results.

Stage5C latent-generative execution and SMC remain disabled. No new deployment,
calibration certificate or submission-readiness claim follows from this round.

## Reading the Repository

| Record | Purpose |
|---|---|
| [Results ledger](README_RESULTS.md) | Detailed outcomes, including negative results and verification status |
| [Research state](research_state.json) | Current machine-readable status and next action |
| [Archived research history](README_RESEARCH_HISTORY_2026_09_27.md) | Complete earlier project overview, preserved before this shorter introduction |
| [Earlier September history](README_RESEARCH_HISTORY_2026_09.md) | Earlier routes and diagnoses |
| [Recording/teacher audit](outputs/publication_readiness_2026_09/recording_lineage_audit.md) | Limits of historical external claims |
| [Data-role contract](outputs/publication_readiness_2026_09/experiment_contract/implementation_and_limits.md) | Training, selection, calibration and confirmation boundaries |
| [Working paper](outputs/publication_readiness_2026_09/paper_working_draft.md) | Research draft, not a finished or submission-ready paper |
| [Local/CREATE runbook](outputs/publication_readiness_2026_09/local_create_runbook_zh.md) | Environment, scheduling and recovery guidance |

Raw datasets, feature/history/latent caches, large checkpoints, videos,
third-party images and local virtual environments stay out of Git.

## Running Locally

On Apple Silicon, use the native arm64 PyTorch environment, not Intel Conda
under Rosetta. Current risk-head experiments use 4 compute threads, 1 interop
thread and 0 DataLoader workers, with atomic checkpoints, heartbeat and resume.

A focused check for the latest experiment:

```sh
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_centered_risk_policy.py tests/test_m3w_centered_risk_verification.py tests/test_m3w_centered_identity_replay.py
```

The reproduction guide above documents the frozen fit, action and evaluation
sequence. Existing sealed outputs are checked, not silently overwritten.
The legacy full test suite includes integration training and can rewrite
reports; it is not an isolated read-only smoke test.

## Next Question

Which conditional errors make useful and harmful interventions hard to separate
across sources? The global-offset experiment now has a negative answer even at
matched counts. I will first inspect easy-risk exposure and error scale within
source-excluded fitting data. Any repair must preserve benefit and easy
protection, retaining an original-score control at the same
intervention count, freezing decisions before readout. No conditional repair
has been trained yet. I will not tune thresholds on these diagnostics or open independent
confirmation data to rescue the method.

The larger goal remains useful neural dynamics with reproducible, independent
evidence. More stages, more overlapping windows or a lower training loss are
not substitutes for that result.
