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
