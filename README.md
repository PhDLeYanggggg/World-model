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

I have registered the next matched bank experiment: keep the same gain/harm
learner, fitting recordings, sampler and risk rule, and change only the frozen
neural candidate. The six physical actions remain unchanged. All 54 fits are
planned across six contexts and three seeds, with the original predictor seed
43 fixed. The new candidate's features, region context and supervised costs
are rebuilt together; this is not a threshold search on the last readout.
[Matched-bank protocol](outputs/publication_readiness_2026_09/european_active_bank_cost_v1/protocol.md).

All 54 cost heads now finish their full 2,000 updates in 4.7 minutes, with
11.0 GB peak memory. The 36 old-bank controls reproduce their complete old
training states, and the first new-bank head has an exact full refit. Every
endpoint is frozen before policy scoring. This checks the controlled training
comparison; it is not yet evidence that the new policy improves utility or risk.

The complete 72-group readout now answers that comparison. Known-label ADE
gain is 1.893% with the new neural bank, versus 1.969% with the old bank and
1.946% with physical actions only. New-minus-old is -0.075pp, CI
[-0.171,+0.012], so replacing the candidate has not improved the policy.
Only 37/444 required recording-risk views pass. Disabling just the neural
action in the same new head removes a small known-label benefit (+0.166pp,
CI [+0.055,+0.334]), but its missing-outcome lower CI crosses zero. I keep
these findings separate: actual neural use can help on known labels without
making the complete learned policy better or safe. All results are frozen
before a separate full inference replay; no deployment is promoted.

That separate replay is complete: all 72 prediction/action/metric groups match
exactly in 2.1 minutes. The new head's actual training-loss median falls from
1.904 to 0.333, but the risk requirement still fails. The
[complete result and curves](outputs/publication_readiness_2026_09/european_active_bank_cost_v1/readout.md),
[failure analysis](outputs/publication_readiness_2026_09/european_active_bank_cost_v1/failure_analysis.md)
and [Chinese reproduction guide](outputs/publication_readiness_2026_09/european_active_bank_cost_v1/reproduction_zh.md)
retain both the small neural-use signal and the complete-policy failure.
My next experiment will address the cost-to-risk decision mismatch, not
promote this bank or tune its thresholds on the same results.

I traced every requested future label on the same 149,510 allowed SOURCE
queries back to the original export. All masks and known coordinates agree,
and a second raw reparse matches exactly. Of 279,192 missing future points,
180,358 follow the last detection of that ID; only 8,948 exceed the record's
last detection frame. This is missing supplied observation support, not an
import omission or proof of physical exit. I have not interpolated new labels
or changed the horizon. [Raw-label audit](outputs/publication_readiness_2026_09/european_source_label_provenance_v1/readout.md).

The integration check also separates two producer families. The 72-model input
study below used the original flat SOURCE backbone; the current routing bank
uses the later grouped, partial-neighbor, dimensionless model. I have registered
the matched detector-input test on all nine actual parent designs, preserving
their masks, units and training schedule. No new trajectory result is claimed
at this registration. [Active-producer protocol](outputs/publication_readiness_2026_09/european_active_detector_forecaster_v1/protocol.md).

The matched active-backbone run now completes all **36 models**, with
**144,000 primary updates**, in 30.4 minutes on the native arm64 CPU runtime
(2.20 GB peak memory). All nine disabled controls reproduce the real parent
models, and the first full history refit is exact. I am freezing every endpoint
before the fixed forecast readout; no accuracy or deployment gain is claimed
at this training checkpoint.

The first fixed active-backbone readout is now complete. History gains 10.486%
known-label ADE over the fitting-selected causal reference; the disabled parent
gains 9.326%. The increment is +1.160pp, CI [-0.355,+2.540], so a stable added
input effect is **not established on the actual active model**. Raw easy-case
accuracy degrades 12.222% relative to that reference, and missing-outcome lower
bounds remain negative. All nine fresh parent controls match exactly. I am
retaining this negative result; the proposal gate fails and deployment is
unchanged. A separate process now reproduces all 36 prediction/metric groups
and all nine fresh parent controls exactly in 9.2 minutes. The real loss curves,
complete locality breakdown and limits are in the
[verified active-backbone readout](outputs/publication_readiness_2026_09/european_active_detector_forecaster_v1/readout.md)
and [failure analysis](outputs/publication_readiness_2026_09/european_active_detector_forecaster_v1/failure_analysis.md).

**Earlier flat-backbone input study (30 October).**
I extended the verified detector-history store to all **318,969 producer rows
across 163 SOURCE recordings**, rather than treating the smaller routing cache
as the whole training set. The raw-input replay agrees exactly. I have trained
**72 matched Transformer forecasters** with the original folds, three seeds,
and all 4,000 updates per model: **288,000 primary updates**, plus a separate
4,000-update full refit and 900 pilot updates. The complete CPU run took
94.5 minutes, with 2.40 GB peak memory. All 18 disabled controls and the first
full history refit reproduce their scientific state exactly.

Every endpoint was frozen before the fixed development readout. Better proposals answer a different
question from a safe switching policy; they would not erase the negative result
below or open the independent evaluation sets.
[Registered experiment](outputs/publication_readiness_2026_09/european_detector_forecaster_v1/protocol.md).

The first complete development readout now gives **8.811% known-label ADE gain**
for detector history, versus **4.923%** without it. The paired change is
**+3.888 percentage points**, locality CI **[+2.246,+5.655]**. History also beats
current-only and identity-noise controls on known labels. But the missing-label
completion-lower contrast is negative, and one required locality has no final
step labels, so the all-locality FDE is undefined. All 72 readout groups now
replay exactly in a separate process. The supplementary paired difference
against fixed damping 0.97 is +2.375pp, CI [-2.731,+6.710], so superiority over
all strong fixed physical baselines is not established. This is promising
observed-label proposal evidence, not a
safe policy, an independent test or a deployment promotion.
[Verified readout and curves](outputs/publication_readiness_2026_09/european_detector_forecaster_v1/readout.md),
[limits and failure analysis](outputs/publication_readiness_2026_09/european_detector_forecaster_v1/failure_analysis.md).

The preceding routing-only experiment used 149,510 fitting queries across
74 SOURCE recordings. Its raw reparse matches exactly, including 222 prefix
checks. I trained **144 matched Torch cost heads**, **288,000 primary updates**,
with disabled, current-only, identity-noise and full-history inputs. All 36
disabled controls reproduce the old endpoints exactly, and all 288 fitting
readouts replay exactly in a separate process.

History improves known-label ADE gain from **0.581% to 1.969%**. The paired
change is **+1.388 percentage points**, fitting locality CI **[+0.413,+2.888]**.
But only **42/444** required recording-risk views pass, with 307 known easy-risk
violations. Adding the neural candidate to the physical bank has no stable
known-outcome increment; its conservative completion-lower CI is negative.
These are fitting results, not independent generalization or a safe deployment.
The extra information has helped some decisions, not yet safe neural dynamics.
Targets, splits and the 2% complete-policy risk requirement are unchanged.
[Complete matched result](outputs/publication_readiness_2026_09/european_detector_cost_v1/readout.md),
[failure analysis](outputs/publication_readiness_2026_09/european_detector_cost_v1/failure_analysis.md).

**29 October: targeted cost training still does not make neural substitution safe.**
I completed eight controlled comparisons over the same frozen forecast bank:
continuous costs, recording-level decision training, a published deferral loss,
support-unit conditioning, shared reference costs, static scene context and
stochastic and complete-region moment training.
Each retains three controller seeds, six contexts and all 54 planned fits.
Together they comprise **432 real Torch heads and 702,000 primary updates**.
These are controller experiments, not 432 new trajectory forecasters. Every
endpoint was frozen before readout, and all eight comparisons replay exactly.

| Mixed-bank policy | Known-label ADE gain over floor | Required risk views supported |
|---|---:|---:|
| Continuous costs | 0.554% | 20/186 |
| Recording-level decisions | 0.000% | 0/186 |
| Published deferral loss | 8.018% | 0/186 |
| Support-unit repair | 0.0348% | 92/186 |
| Shared reference | 0.542% | 21/186 |
| Static scene context | 0.983% | 23/186 |
| Policy-region moments | 0.944% | 19/186 |
| Complete-region gradients | 0.944% | 24/186 |
| Matched conventional cost forest | -0.000073% | 118/186 |

The preceding feature ablation adds 18 static/past/rollout region features using
14 correctly matched camera polygons. Relative to the no-scene control, known-
label utility improves **0.429 percentage points**, conditional 95% locality
CI **[+0.054,+1.045]**. Harm-cost MSE also decreases. This is evidence that
image-region position matters for cost prediction, not a deployment result.

The physical-only scene control gains **1.077%**, more than its mixed
policy's **0.983%**. The paired neural increment is **-0.0938pp**, CI
**[-0.2179,-0.0034]**. Missing-outcome completion bounds do not establish a
positive scene gain, and the mixed policy selects 518 unknown-outcome row views.
Lower overall prediction error has not controlled risk after action selection.

The TRAIN-only audit now exactly replays all 108 original/scene cost endpoints.
Easy-harm bias is concentrated inside the gate despite much better all-legal
calibration. I then trained 54 matched heads with an aggregate moment loss on
the old policy's selected regions. The new mixed policy gains **0.944%**, but
its change from the scene control is **-0.039pp**, CI **[-0.275,+0.135]**.
Its neural increment over its physical-only bank is **+0.039pp**, CI
**[-0.087,+0.166]**; the missing-outcome completion-lower contrast is negative.
Only **19/186** required risk views pass. This repair does not earn deployment.

The fitting diagnosis finds incomplete easy-harm calibration even inside the
old regions. I also checked the optimization: minibatch and complete-region
gradients differ substantially, and some small regions never receive the
auxiliary loss. A controlled repair processes complete regions in chunks.
It completes all 54 fits and replays exactly, but known utility changes by only
**-0.00008pp**, CI **[-0.108,+0.097]**. Its neural increment is **+0.011pp**,
CI **[-0.122,+0.102]**, with just **24/186** required risk views supported.
Even fixed-region easy-harm fitting remains poor. The optimization repair
does not earn deployment.

This experiment takes 108,000 updates and 12,275,011 additional teacher-row
visits. Those are repeated fitting visits, not new data or equal training FLOPs.
The 186 risk views repeat contexts and head seeds over 31 unique recordings.
Empty selected risk is undefined, never a passed safety check; count matching
concerns utility only. Earlier pointwise-deviance and recalibration failures
remain part of the record.

I then separated candidate opportunity from learned routing. On the exposed
inner-held data, a future-informed physical oracle gains **32.40%** over the
floor; adding the frozen neural path raises that to **33.90%**. Neither is
deployable, and their difference does not bound differences between causal
learned policies.

I also refit **54 conventional cost forests (6,912 trees)** on the current
seven-action task, using the same causal information, fitting normalizers and
query draws as the static-scene neural cost heads. The mixed forest's gain is
**-0.000073%**: effectively no useful improvement. Its change from the neural
cost head is **-0.983pp**, conditional 95% CI **[-2.302,-0.242]**. Interventions
fall from 32,219 to 3,457 repeated row views. Risk support improves to 118/186,
but 66 easy-risk views remain undefined and two have known violations. This
is a more conservative policy, not a successful repair. The objectives and
decoders differ, so the result does not isolate architecture or rule out every
conventional estimator. Training took 40.9 minutes, and all 144 readout groups
replay exactly. [Verified results](outputs/publication_readiness_2026_09/european_multiaction_forest_v1/readout.md).

The paired fitting audit sharpens the remaining problem. All-legal aggregate
easy-harm ratios are near one for both estimators, but within identical frozen
neural-selected regions the forest predicts 4.576 times the observed easy harm
and the neural head only 0.0214 times. This is not a calibrated risk interval.
The subsequent fitting-only bank control now replays all 144 groups exactly.
Most forest suppression occurs in the estimated risk gate: 85,117 dedicated-
expert passes are rejected by the mixed estimator, whereas only 166 mixed-
gate passes lose to a physical competitor. Separating the experts recovers
some fitting utility, but required recording risk support falls from 330 to
285/444 for forests and from 77 to 57/444 for neural cost heads. It does not
repair safety. No new model is trained, and no held or independent role is
opened. The useful next question concerns safer candidate outputs, not merely
more permissive switching.

I then tested five fixed output amplitudes for both physical and neural
candidates, with and without changing the original selection set. All 144
fitting groups and 720 amplitude views complete and replay exactly. With
the neural cost heads, shrinking the mixed policy to 1/16 on its original
set reduces known easy-risk violations to zero and raises risk support from
77 to 291/444, but gain falls from 0.678% to 0.047%. Reselecting at that
amplitude gains 0.819%, yet supports only 40/444 views and selects many more
unknown outcomes. No setting passes complete safety. Smaller predictions
are not enough when the selected region changes or lacks outcome support.
These are fitting controls, not newly trained dynamics or independent gains.

The next paired training comparison includes conservative missing-outcome
cost endpoints in the loss, against a fresh known-only control on the same
causal query draws. The original unknown labels stay unknown. This tests a
training/evaluation mismatch, not a new forecast architecture or a change to
the safety requirement. [Fixed experiment](outputs/publication_readiness_2026_09/european_completion_cost_v1/protocol.md).

All 72 new cost heads finish their fixed 2,000-step runs and are frozen
before readout. This is real Torch optimization, not new trajectory-forecast
training. Each supervision mode sees the same 348,388 repeated unknown-row
draws; only the completion mode includes their interval endpoints in the
loss. The completed fitting readout is negative: mixed-policy known ADE gain changes
from +0.581% for the new known-only control to -18.512% with completion
endpoints. Required recording risk support falls from 77 to 26/444. The
paired change is -19.093pp, fitting 95% locality interval [-50.820,-1.022].
All 144 groups replay exactly. The subsequent 888-view fixed-region diagnosis
also replays exactly: the new estimator improves easy-harm estimation on the
old policy's selected region but severely underestimates it on its own new
region. Valid interval targets do not by themselves make a learned gate safe.

A reject-only patch is insufficient here. The old policy lacks selected
easy-reference support in 62/444 views; deleting actions cannot create that
support. This restricts one repair class, not every possible model. I retain
the negative result, its actual loss traces and both candidate-bank controls.
The next method must address risk estimation on newly admitted regions, not
simply make this failed head more conservative.

I also tested coupled nearest-neighbor cost estimates using other fitting
recordings as donors. All 288 fixed comparisons and a separate full replay
complete. The primary gains **0.830%** on known-label ADE but supports only
**30/444** required risk views. Its neural-bank increment has a small positive
fitting interval, **+0.0409pp [0.0012,0.0866]**, while the full policy remains
unsafe. A global mean is more conservative but gains only 0.004%. Consistent
cost outputs and higher average accuracy have not solved the selection problem.

A further control preserves donor recording and action-availability profiles
while shuffling their fine target associations. The real matching advantage
has an interval crossing zero. I am now checking the camera inputs before
trying another learner: supplied control points exist for all 14 fitting
cameras, but static reprojection residuals and coordinate conventions still
need resolution. Their existence is not verified metric calibration.

These are exposed European SOURCE-development results, not independent
confirmation or whole-pipeline OOF. The data are oblique webcam detector-track
image coordinates, not SDD's official top-down annotations. The [data card](outputs/publication_readiness_2026_09/european_shared_reference_cost_v1/data_card.md)
distinguishes the publisher's release, our adapter and the experimental cohort.
Static polygons are region proxies, not verified walkable or metric maps.
Independent evaluation stays closed; submission readiness is still false.
Metric/effective-second claims, Stage5C execution and SMC remain off.

- [Six-study comparison, intervals and moment errors](outputs/publication_readiness_2026_09/feature_cost_evidence_20261029.md)
- [Seventh comparison: region-moment evidence](outputs/publication_readiness_2026_09/european_policy_region_cost_v1/evidence.md)
- [Eighth comparison: complete-region evidence](outputs/publication_readiness_2026_09/european_balanced_region_cost_v1/evidence.md)
- [Exact-gradient repair and failure analysis](outputs/publication_readiness_2026_09/european_balanced_region_cost_v1/readout.md)
- [New manuscript evidence addendum](outputs/publication_readiness_2026_09/evidence_addendum_complete_regions_20261029.md)
- [Neural candidate opportunity and its causal interpretation](outputs/publication_readiness_2026_09/european_neural_marginal_ceiling_v1/readout.md)
- [Matched conventional control, fixed-region diagnosis and limits](outputs/publication_readiness_2026_09/european_multiaction_forest_v1/readout.md)
- [Conventional-control failure analysis and paper addendum](outputs/publication_readiness_2026_09/european_multiaction_forest_v1/paper_addendum.md)
- [Bank-factorization control: exact gate attribution and its safety failure](outputs/publication_readiness_2026_09/european_cost_bank_factorization_v1/readout.md)
- [Fixed forecast-amplitude controls and complete negative results](outputs/publication_readiness_2026_09/european_forecast_amplitude_v1/readout.md)
- [Amplitude tradeoff figure](outputs/publication_readiness_2026_09/european_forecast_amplitude_v1/figures/amplitude_controls.svg)
- [Current availability of matched SOURCE image inputs](outputs/publication_readiness_2026_09/source_rgb_availability_20261029.md)
- [Completion-cost training: full results, intervals and real losses](outputs/publication_readiness_2026_09/european_completion_cost_v1/readout.md)
- [Selection-region failure and restricted support argument](outputs/publication_readiness_2026_09/european_completion_cost_v1/failure_analysis.md)
- [Completion-cost comparison figure](outputs/publication_readiness_2026_09/european_completion_cost_v1/figures/policy_comparison.svg)
- [Completion-cost training curves](outputs/publication_readiness_2026_09/european_completion_cost_v1/figures/loss_traces.svg)
- [Completion-cost local reproduction guide](outputs/publication_readiness_2026_09/european_completion_cost_v1/reproduction_zh.md)
- [Coupled local-cost comparison and complete negative safety result](outputs/publication_readiness_2026_09/european_coupled_local_cost_v1/readout.md)
- [Coupled cost-control figure](outputs/publication_readiness_2026_09/european_coupled_local_cost_v1/figures/control_comparison.svg)
- [Stratified matching control and its limits](outputs/publication_readiness_2026_09/european_coupled_local_cost_v1/stratified_readout.md)
- [Training-camera projection resource audit](outputs/publication_readiness_2026_09/source_projection_audit_v1/readout.md)
- [Region-moment failure and fitting mechanism](outputs/publication_readiness_2026_09/european_policy_region_cost_v1/readout.md)
- [Static scene ablation and failure analysis](outputs/publication_readiness_2026_09/european_static_scene_cost_v1/readout.md)
- [Support-unit repair and failure analysis](outputs/publication_readiness_2026_09/european_multiaction_support_scale_v1/readout.md)
- [Published-loss control](outputs/publication_readiness_2026_09/european_regression_deferral_v1/readout.md)
- [Shared-reference result and limitations](outputs/publication_readiness_2026_09/european_shared_reference_cost_v1/readout.md)
- [Static scene-context audit](outputs/publication_readiness_2026_09/european_static_scene_audit_v1/readout.md)
- [English research draft, revision 7](outputs/publication_readiness_2026_09/evidence_manuscript_v7/README.md)
- [Frozen four-study evidence and revision 6](outputs/publication_readiness_2026_09/evidence_manuscript_v6/README.md)
- [Chinese operating guide](outputs/publication_readiness_2026_09/evidence_manuscript_v6/reproduction_zh.md)

### Completed Harm-Loss Comparison

**29 October: penalizing forecast harm helps some easy cases, but does not improve the model.**
I tested a loss that penalizes error increases over the training-selected
baseline and causal constant velocity (CV). Architecture, data, folds, seeds
and the 4,000-update training schedule were fixed within each comparison.
All **27 neural fits and 108,000 primary updates** are complete: nine fits on
the actual current producer, plus a separate 18-fit earlier-family comparison.
Both pass exact neutral and interruption-recovery checks. Every endpoint was
frozen before readout, and both result sets replay exactly.

For the current producer, equal-locality ADE improvement over CV falls from
**8.46% to 7.53%**. Easy ADE degradation improves from 11.17% to 8.43%, but is
still too high; hard improvement falls from 13.36% to 11.68%. The paired overall
change is **-0.93 percentage points**, with a conditional 95% locality-bootstrap
interval of **[-1.67, -0.17]**. All three seeds regress. These are 8-observed /
12-predicted European SOURCE development results, not historical `t+50` scores
or independent confirmation.

The fixed-floor TRAIN diagnostic gains one feasible recording/context view:
**147/148 instead of 146/148**. It still fails on one required view. A separate
geometry check identifies an actionable restriction: the fold-2 neural output
cannot represent CV on **43,743 evaluated windows**. That does not explain the
remaining difficult recording, where CV is geometrically representable.

The current and earlier-family training took 11.3 and 27.3 minutes respectively.
54 targeted regression tests pass; the full historical suite was not rerun.
I am keeping deployment and the 2% full-policy safety requirement unchanged.
That output-construction comparison is now complete above; its small gain does
not reverse the earlier harm-loss failure.
[Results](outputs/publication_readiness_2026_09/european_dimensionless_harm_forecast_v1/conclusions.md),
[failure analysis](outputs/publication_readiness_2026_09/european_dimensionless_harm_forecast_v1/failure_analysis.md),
[output-bound diagnosis](outputs/publication_readiness_2026_09/european_dimensionless_harm_forecast_v1/output_bound_diagnosis.md),
[evidence gate](outputs/publication_readiness_2026_09/european_dimensionless_harm_forecast_v1/evidence_gate.md),
[reproduction guide](outputs/publication_readiness_2026_09/european_dimensionless_harm_forecast_v1/reproduction_zh.md).
The [lineage clarification](outputs/publication_readiness_2026_09/european_reference_harm_forecast_v1/lineage_clarification.md)
and [earlier-family negative result](outputs/publication_readiness_2026_09/european_reference_harm_forecast_v1/conclusions.md)
are retained separately.

### Previous Recording-Constraint Comparison

**29 October: balancing recording constraints improves fitting coverage, but the policy remains unsafe.**
I trained 18 neural policy heads for 36,000 updates with equal recording weights
for constraint penalties, leaving the benefit objective and the 2% budgets
unchanged. Required recording-budget coverage increases from **84/444 to
131/444 views**, with improvement in all three policy-head seeds. The policy
retains **80.93% of eligible positive benefit**, versus 81.55% for its control.

This is only a partial repair. There are still **289 known all-harm violations**;
full-policy utility falls slightly even though matched-count utility improves.
The 444 views repeat 74 unique recordings across contexts and seeds. These are
fitting results, not independent generalization or a deployment safety claim.
The known fixed-forecast support obstruction also remains unresolved.

All 18 final weights were frozen before readout. Training took 12.2 minutes
locally with 2.82 GB peak memory; all 72 readouts replay exactly, and 68 regression
tests plus three new analysis tests pass. Independent evaluation remains unopened.
I am keeping deployment unchanged and prioritizing candidate forecast/support
repair rather than another threshold search.
[Results](outputs/publication_readiness_2026_09/european_record_balanced_policy_v1/conclusions.md),
[failure analysis](outputs/publication_readiness_2026_09/european_record_balanced_policy_v1/failure_analysis.md),
[model card](outputs/publication_readiness_2026_09/european_record_balanced_policy_v1/model_card.md),
[reproduction guide](outputs/publication_readiness_2026_09/european_record_balanced_policy_v1/reproduction_zh.md).

### Previous Cost-Sensitive Comparison

**29 October: the new loss recovers useful interventions, but safety still fails.**
I trained 18 cost-sensitive neural policy heads for 36,000 updates, changing
only the primal loss. On the fitting data, they retain **81.55% of eligible
positive benefit**, compared with 26.00% for the frozen exact-gradient control.
At the same intervention count, the utility difference is bounded between
**+1.8240 and +1.8248 percentage points** of reference cost. These are TRAIN
outcome-completion bounds, not an FDE gain, confidence interval or test result.

The required recording-level safety result worsens: **84/444 views pass,
versus 178/444**. Most known harm violations occur in two multi-recording
sources. Preserving net easy error does not satisfy the separate positive-harm
constraints. This is a useful fitting improvement, not a deployable policy.

All final weights were frozen before readout; all 72 readouts replay exactly,
and 57 targeted tests pass. Independent evaluation remains unopened. I am
keeping deployment unchanged. This improved benefit-learning loss motivated
the completed recording-constraint comparison above. The known forecast/support
obstruction remains visible in both experiments; neither is a deployment update.
[Results](outputs/publication_readiness_2026_09/european_cost_sensitive_policy_v1/conclusions.md),
[failure analysis](outputs/publication_readiness_2026_09/european_cost_sensitive_policy_v1/failure_analysis.md),
[model card](outputs/publication_readiness_2026_09/european_cost_sensitive_policy_v1/model_card.md).
[Follow-up protocol](outputs/publication_readiness_2026_09/european_record_balanced_policy_v1/protocol.md).

### Previous TRAIN Feasibility Diagnostic

**29 October: safe training choices exist, but the policy does not recover them.**
An exact label-aware diagnostic finds useful choices within the 2% budget in
146 of 148 recording/forecast views. The two exceptions are the same recording:
its fixed forecast has no eligible easy-reference choice within budget.
The cross-record neural policy retains only 23.46% of the eligible positive benefit.
Its selected set contains a safe useful subset in 371 of 444 record/seed views, while
the actual policy passes in 160. This points to both decision-fitting failure and
a specific forecast/support limitation, not simply an absence of useful signal.

All 72 fitting readouts replay exactly. An initial grouping bug is preserved and
documented; corrected per-record scores match the prior verified controls.
The oracle is label-aware and not deployable. It motivated the cost-sensitive
logistic comparison reported above, with the same inputs and 2% budget.
Its 18 full fits took 10.1 minutes with 3.02 GB peak memory; no held data or
deployment was changed.
[Diagnosis](outputs/publication_readiness_2026_09/european_train_feasibility_v2/conclusions.md),
[follow-up protocol](outputs/publication_readiness_2026_09/european_cost_sensitive_policy_v1/protocol.md).

### Previous Cross-Recording Comparison

**29 October: cross-recording meta-training did not repair policy safety.**
I trained 36 neural intervention heads for 72,000 updates, testing whether an
update learned on one group of recordings remains useful on another group.
The comparison used matched same-record training, fixed forecasts and features,
and the unchanged 2% safety requirement. All final weights were frozen before
evaluation. Training took 58.5 minutes locally, with about 3.7 GB peak memory.

The primary policy meets the budget in **51/72 fitting views but 0/72 held views**.
It retains about **47% of the original policy's known benefit and 47% of its
known harm**: the desired separation has not emerged. At equal intervention
counts, its utility change against the same-record control is bounded between
**-0.189 and -0.045 percentage points** of reference cost. This is a partial-label
completion bound, not an FDE improvement or confidence interval.

A 3,000-draw source bootstrap gives an outer envelope of **[-0.454, +0.063] pp**
for that matched contrast. It provides no evidence of improvement, but also
does not establish that cross-record meta-learning is generally worse. The
complete-policy safety failure is clear, so deployment remains unchanged.

All 144 readouts replay exactly and 113 targeted tests pass. These are exposed
TRAIN recordings, not independent confirmation; eight sources have only one
fitting recording. My next step is to separate budget/support infeasibility from
optimizer failure on TRAIN before trying another policy objective.
[Results](outputs/publication_readiness_2026_09/european_cross_record_policy_v1/conclusions.md),
[complete tables](outputs/publication_readiness_2026_09/european_cross_record_policy_v1/readout.md),
[failure analysis](outputs/publication_readiness_2026_09/european_cross_record_policy_v1/failure_analysis.md),
[reproduction](outputs/publication_readiness_2026_09/european_cross_record_policy_v1/reproducibility.md).

### Previous Whole-Recording Comparison

**29 October: whole-recording gradients improve fitting, but risk does not transfer.**
I trained 36 neural intervention heads with 72,000 updates, comparing exact
recording-level constraint statistics and full-recording gradients with the
previous stochastic method. The forecasts, features, update count and 2% budget
were fixed. Full gradients use more computation; this is not an efficiency claim.

The primary model meets the safety budget in **53/72 training views**, up from
35/72, but **0/72 held-recording views**, down from 4/72. Its held utility improves
over the stochastic method, yet at the same intervention count it still loses
to the original frozen policy. That comparison is bounded between **-1.518 and
-0.628 percentage points** of reference cost, not a confidence interval or an
FDE improvement. Better fitting has not produced a safe policy.

The diagnostic did identify a real sampling problem: small batches frequently
miss positive risk excess. Correcting that alone was insufficient. The new
policy retains about **53% of the original known benefit and 53% of its known
harm** on held recordings. I am keeping deployment unchanged and focusing the
next experiment on recording-level constraint generalization, not threshold tuning.

All 144 readouts replay exactly and 100 targeted tests pass. These checks support
execution, not independent scientific confirmation: the recording holdout was
previously exposed, and independent evaluation remains unopened.
[Results](outputs/publication_readiness_2026_09/european_whole_record_policy_v1/conclusions.md),
[complete comparisons](outputs/publication_readiness_2026_09/european_whole_record_policy_v1/readout.md),
[failure analysis](outputs/publication_readiness_2026_09/european_whole_record_policy_v1/failure_analysis.md),
[reproduction](outputs/publication_readiness_2026_09/european_whole_record_policy_v1/reproducibility.md).

### Previous Direct Policy

**29 October: direct decision learning reduces some harm, but loses useful interventions.**
I trained 54 neural intervention heads with 108,000 optimizer updates, comparing
source-level constraints, recording-level constraints and hard deployment
decisions during training. The trajectory predictors and comparison protocol
were unchanged. Every final model was frozen before the recording-holdout readout.

The primary hard-decision model meets the safety budget in **35/72 training
views**, but only **4/72 held-recording views**. It retains about **21% of the
original policy's known benefit and 20% of its known harm** on held recordings.
At the same intervention count, its utility change is bounded between
**-1.133 and -0.625 percentage points**. These are partial-label completion
bounds, not confidence intervals or FDE improvements. Fewer interventions alone
do not explain the failure: the retained choices are also less useful.

Hard decisions modestly improve matched utility over the soft recording-level
version, but neither beats the original policy or satisfies the full safety
requirement. I am keeping deployment unchanged. The next question is whether
whole-recording optimization can preserve useful choices while actually meeting
the training constraints, before claiming transfer to other recordings.

This is new neural **policy** training, not a new trajectory forecaster. Exact
replay and 88 targeted tests support the implementation; earlier exposure makes
the recording holdout exploratory, not independent confirmation.
[Results](outputs/publication_readiness_2026_09/european_direct_policy_v1/conclusions.md),
[complete comparisons](outputs/publication_readiness_2026_09/european_direct_policy_v1/readout.md),
[failure analysis](outputs/publication_readiness_2026_09/european_direct_policy_v1/failure_analysis.md),
[reproduction](outputs/publication_readiness_2026_09/european_direct_policy_v1/reproducibility.md).

### Previous Easy-Constrained Check

**29 October: preserving the training objective does not yet preserve policy safety.**
I completed 72 comparisons of an easy-constrained risk estimator and evaluated
the resulting intervention decisions. The new objective removes 16,261 donor-fit
conflicts where total loss improved at the expense of easy cases. But the complete
policy meets the unchanged safety budget in only **5/72** fitting groups.
Even on known labels, 33 groups exceed the overall budget and 22 exceed the easy
budget. Missing outcomes are part of the problem, not the whole explanation.

At the same intervention count, the new estimator improves the utility lower
bound by **0.098 percentage points** over the previous repair, but remains
**0.692 points below the original policy**. These are partial-label utility
bounds, not confidence intervals or trajectory-accuracy gains. I am retaining
all six comparisons and keeping deployment unchanged.

This test evaluates decisions directly; risk MSE is a secondary diagnostic, not
an additional hurdle. The next method question is whether learning intervention
utility and scene-level risk directly works better than fitting risk by squared
error. The current results are exposed-TRAIN evidence, not independent testing.
[Results and diagnosis](outputs/publication_readiness_2026_09/european_easy_constrained_v1/conclusions.md),
[complete tables](outputs/publication_readiness_2026_09/european_easy_constrained_v1/readout.md),
[protocol](outputs/publication_readiness_2026_09/european_easy_constrained_v1/protocol.md).

### Previous Joint Risk Check

**29 October: joint risk estimates are coherent, but still not accurate enough.**
I completed and exactly replayed all 72 fitting comparisons with unchanged
queries and neighbors. Joint constraints remove the 9,019 contradictory outputs
where easy harm exceeded total harm. Donor-record-excluded blending lowers easy
risk-estimation error by **26.09%** versus the previous cost fit, but it remains
**5.69% worse than the original risk head**. Only **1/18** paired comparisons
beats both controls; the registered screen fails.

The learned per-query blend also loses to using one mean blend weight within
each recording. Reference concentration is real, but does not fully explain
the failure. Shrinking predictions repairs part of their magnitude, not the
ability to locate rare harmful cases. I am keeping deployment unchanged.

These are exposed-TRAIN risk errors, not trajectory ADE/FDE gains or independent
generalization results. No new neural forecaster was trained. Full replay,
separate receipt checks and 79 targeted tests support reproducibility, not a
successful world-model claim.
Future policy evaluations keep full-policy safety separate from count-matched
utility comparisons. Empty risk comparisons remain not estimable; the 2% budget
and historical failed verdicts are unchanged.
[Results](outputs/publication_readiness_2026_09/european_coupled_support_v1/conclusions.md),
[failure analysis](outputs/publication_readiness_2026_09/european_coupled_support_v1/failure_analysis.md),
[gate](outputs/publication_readiness_2026_09/european_coupled_support_v1/evidence_gate.md),
[reproduction](outputs/publication_readiness_2026_09/european_coupled_support_v1/reproducibility.md).

### Previous Local Cost Check

**29 October: aligning the local loss helps, but the risk estimates are still not usable.**
I completed the 72-view cost-objective comparison with the same selected queries
and nearest histories. Direct cost fitting reduces easy-risk prediction error
by **81.30%** against uniform neighbors, but remains **43.00% worse** than the
original risk head. None of the 18 context-seed pairs beats both controls.
These are fitting risk-estimation errors, not trajectory ADE/FDE improvements.

The controls narrow the problem: cost weighting explains almost all of the
improvement. Reference-error offsets add little and produce **9,019** logically
inconsistent predictions where easy harm exceeds total harm. Some nominal
32-neighbor fits also depend almost entirely on one reference sample. I am not
deploying this estimator or opening held data to rescue the result.

The first diagnostic-writer failure is preserved and repaired. The complete
calculation and numerical replay now finish successfully; 74 related tests pass.
No new neural trajectory model was trained. The next work concerns coherent
joint risk estimates and reliable reference support, not another threshold sweep.
[Results](outputs/publication_readiness_2026_09/european_local_cost_analogue_v2/conclusions.md),
[failure analysis](outputs/publication_readiness_2026_09/european_local_cost_analogue_v2/failure_analysis.md),
[gate](outputs/publication_readiness_2026_09/european_local_cost_analogue_v2/evidence_gate.md),
[reproduction](outputs/publication_readiness_2026_09/european_local_cost_analogue_v2/reproducibility.md).

### Previous Reference-Population Check

**29 October: matching the reference population helps, but does not make risk reliable.**
I completed 72 fitting comparisons using reference samples selected by the same
frozen policy as the queries. Query selection, normalization and neighbor settings
were unchanged, and all 62,902 query occurrences retained reference support.

This reduces easy-risk prediction error by **50.10%** and overall risk error by
**65.79%** relative to the previous pooled-neighbor estimate. However, easy-risk
error remains **7.65 times** the original risk head's, and none of the 18
context-seed pairs beats both controls. The registered magnitude screen fails.
These are risk-estimation errors, not trajectory ADE/FDE improvements.

The diagnosis is more specific now: matching average event frequency does not
recover which samples incur harm or how much. Predicted easy harm remains
**3.88 times** observed. I am retaining the negative result and the existing
deployment policy; no new neural model or held-data evaluation was run. Full
numerical replay, separate arithmetic checks and 53 related tests support the
implementation, not an independent generalization claim.
[Results](outputs/publication_readiness_2026_09/european_selection_conditioned_analogue_v1/conclusions.md),
[failure analysis](outputs/publication_readiness_2026_09/european_selection_conditioned_analogue_v1/failure_analysis.md),
[evidence gate](outputs/publication_readiness_2026_09/european_selection_conditioned_analogue_v1/evidence_gate.md),
[reproduction](outputs/publication_readiness_2026_09/european_selection_conditioned_analogue_v1/reproducibility.md).

### Previous Pooled-Reference Check

**29 October: reference coverage is repaired, but risk magnitude is not.**
I completed all 72 fitting comparisons after pooling four approved sources from
the same predictor context. Excluding the query recording still leaves enough
donors for every one of the 62,902 query occurrences. No held or independent
data were added, and no new neural model was trained.

Nearest histories reduce easy-risk prediction error by **14.99%** against pooled
means and **58.17%** against shuffled labels. The registered information screen
passes in 12/18 context-seed pairs. But the error remains **15.32 times** the
original risk head's, and predicted easy harm is **7.30 times** observed. Overall
risk error also worsens. This is evidence of fitting information, not a safe
replacement model; deployment remains unchanged.

The original hash-format pilot failure is preserved and repaired without changing
the scientific rule. Full results, source slices and verification are documented
below. These are exposed-TRAIN diagnostics, not whole-pipeline out-of-fold or
independent generalization evidence.
[Results](outputs/publication_readiness_2026_09/european_pooled_analogue_v2/conclusions.md),
[failure analysis](outputs/publication_readiness_2026_09/european_pooled_analogue_v2/failure_analysis.md),
[evidence gate](outputs/publication_readiness_2026_09/european_pooled_analogue_v2/evidence_gate.md),
[reproduction](outputs/publication_readiness_2026_09/european_pooled_analogue_v2/reproducibility.md).

### Completed Source-Local Analogue Check

**29 October: similar histories carry some information, but risk is still unreliable.**
I completed a 72-view check using past-only features from other recordings to
estimate harm on the original selected cohort. Only 24 views had complete
reference support: in eight source localities, excluding the query recording
left no eligible donor recording. Those unsupported cases remain in the primary
result, which is undefined rather than silently restricted to successful sources.

Within the 24 supported views, nearest histories reduced easy-risk error by
58.42% against a source mean and 74.53% against shuffled-label neighbors. Yet
their error was still 3.90 times the original risk head's, and predicted easy
harm was 5.48 times observed. Local feature information is not calibrated risk.
The registered screen did not pass, and I am not changing deployment.

All 72 views reproduce exactly; separate arithmetic and 24 related tests pass.
The calculation took 160.68 seconds and 2.92 GB peak memory. It involved no new
neural training or independent evaluation. These source recordings already
have upstream training exposure, so reference exclusion is not whole-pipeline
out-of-fold evidence. [Results](outputs/publication_readiness_2026_09/european_selected_analogue_v1/conclusions.md),
[failure analysis](outputs/publication_readiness_2026_09/european_selected_analogue_v1/failure_analysis.md),
[reproduction](outputs/publication_readiness_2026_09/european_selected_analogue_v1/reproducibility.md).

### Completed Risk-Objective Comparison

**29 October: better average risk fitting still does not make selection safe.**
I completed 36 matched continuations from 18 frozen risk heads. Removing the
action-dependent losses reduced average fitting all/easy risk error by
**80.29% / 55.65%**, while benefit and reference predictions stayed fixed.
But easy-risk error on the original selected cohort did not improve, and only
4 of 18 context-seed pairs improved over both controls. The registered screen
failed, so this candidate does not advance to a new held evaluation.

The unchanged decision rule selected far more occurrences, from 62,902 to
419,270, and produced known easy-risk violations in **62/72 views**. These are
exposed TRAIN results, not independent samples or a deployment success. The
result shows why I cannot equate a lower average training error with reliable
intervention risk.

The 72,000 real updates took 370.46 seconds and 2.90 GB peak memory. All 72
fitting views reproduce from frozen models; separate arithmetic checks and 32
related tests pass. Deployment, independent data roles and the risk criteria
are unchanged. [Results](outputs/publication_readiness_2026_09/european_risk_decoupling_v1/conclusions.md),
[failure analysis](outputs/publication_readiness_2026_09/european_risk_decoupling_v1/failure_analysis.md),
[reproduction](outputs/publication_readiness_2026_09/european_risk_decoupling_v1/reproducibility.md).

### Completed Conditioning Diagnosis

**29 October: lower training loss is not yet better switching-risk estimation.**
I completed a fixed-cohort diagnosis of 36 frozen risk heads. Across 288 matched
batches, removing clipping from a disposable AdamW step more often lowers total
loss (249 batches), but improves the key easy-risk error in only 82, worsens it
in 74 and leaves 132 tied. On the same fitting selection, both models still
underestimate easy harm; peer context improves aggregate mass but worsens
per-row signed-risk error. I am not deploying an unclipped model from this test.

The run includes 72 fitting views and 576 disposable updates, not 576 newly
trained models. No held readout or independent data role was opened. A float32
verification issue was reproduced and repaired using double-precision checks;
the original model, optimizer and completed numerical receipts are unchanged.
Thirty related tests pass. [Results and limitations](outputs/publication_readiness_2026_09/european_query_conditioning_v2/conclusions.md).

### Completed Context Comparison

**29 October: pooled contemporaneous context did not improve safe selection.**
I completed 36 matched risk-head fits and 72 evaluation views. The new head sees
other agents available at the same instant; its self-only control has the same
parameter count, initialization, samples and training budget. All eight existing
controls reproduce. At the same intervention count, peer context reduces utility
by **0.239 percentage points**, nominal 95% CI **[-0.353, -0.144]**. All three
head seeds are negative. Complete risk support remains **0/72**, so I am not
deploying this model.

The failure is not just fewer interventions: matched decisions lose some known
benefit and add some known harm. Unknown outcomes further weaken the evidence.
This rejects the tested mean-pooling construction, not interaction modeling in
general. The data remain exposed original-TRAIN recordings, not independent
confirmation; three risk-head seeds are not three new forecaster trainings.

Before readout, I caught and repaired an accidentally frozen copied encoder.
The original run is preserved; the corrected run verifies actual encoder updates,
exact resume and protected base outputs. Full inference and statistics replay;
60 scoped tests pass. Deployment and reserved data roles remain unchanged.
[Results](outputs/publication_readiness_2026_09/european_query_context_v2/conclusions.md),
[failure analysis](outputs/publication_readiness_2026_09/european_query_context_v2/failure_analysis.md),
[reproduction](outputs/publication_readiness_2026_09/european_query_context_v2/verification.md).

### Preceding Conditional Calibration

**29 October: conditional risk calibration still loses useful interventions.**
I completed72 source-excluded comparisons with frozen neural heads. Conditional
correction increases fully supported risk views from0/72 to20/72, but reduces
selected occurrences from27,371 to5,693. At the same intervention count, utility
falls by **0.366 percentage points**, nominal95%CI **[-0.705,-0.109]**.
It also loses to global calibration and a support-only control.

The problem is not just coverage: useful decisions are removed, and coarse
cell calibration can fail again after selection within that cell.33/72 views
have undefined easy risk. This is exposed original-TRAIN mechanism evidence,
not a new independent result or deployment success. I keep the deployment
unchanged and will not repeat the same calibration sweep with new thresholds.
[Results](outputs/publication_readiness_2026_09/european_policy_conditional_v1/conclusions.md),
[failure analysis](outputs/publication_readiness_2026_09/european_policy_conditional_v1/failure_analysis.md),
[protocol](outputs/publication_readiness_2026_09/european_policy_conditional_v1/protocol.md).

### Preceding Rational Repair

**29 October: removing flat gradients did not restore safe selection.**
I completed18 matched rational-map continuations and144 evaluation views.
On the same fitting cohort, exact-zero easy-harm gradients fell from92.4% to
0.39%. The numerical repair worked, but selected-risk estimation still failed.

At matched intervention counts, utility relative to the hard-clipped model
fell by **0.217 percentage points**, with nominal95%CI **[-0.377,-0.062]**.
Complete risk support remains0/72 held views;19/72 have undefined easy risk.
Fewer known violations cannot be treated as improved safety when the chosen
cohort and undefined denominators change. I am not deploying this repair.

All results remain exposed TRAIN-internal evidence.119 focused tests pass;
no independent data role or risk criterion changed. The next step is
producer-excluded, policy-conditional calibration, not another ad hoc output map.
[Results](outputs/publication_readiness_2026_09/european_rational_harm_v1/conclusions.md),
[failure analysis](outputs/publication_readiness_2026_09/european_rational_harm_v1/failure_analysis.md),
[gradient verification](outputs/publication_readiness_2026_09/european_rational_harm_v1/gradient_recheck/conclusions.md).

### Preceding Wider-Range Experiment

**29 October: expanding the harm head's range did not make it safe.**
I completed18 matched continuations and144 evaluation views after diagnosing
the previous branch's limited ability to correct near-zero harm estimates.
The new mapping preserves causal bounds and the frozen benefit/reference
outputs, with identical initialization, sampling and training budget.

At the same intervention count, held utility versus the old isolated branch
changed by **+0.243 percentage points**, but its nominal95% interval spans
**[-0.057, +0.591]**. Signed-score prediction error worsened, and complete
policy support fell from3/72 views to0/72. This repair does not advance.
It also produced much stronger gradient clipping, so wider expressivity and
optimization conditioning cannot be treated as separate explanations yet.

The experiment ran normally and all controls reproduced. It remains exposed
TRAIN-internal mechanism evidence, not independent forecasting or deployment
success. No risk criterion, reserved data role or deployment was changed.
[Results](outputs/publication_readiness_2026_09/european_full_range_harm_v1/conclusions.md),
[failure analysis](outputs/publication_readiness_2026_09/european_full_range_harm_v1/failure_analysis.md),
[model card](outputs/publication_readiness_2026_09/european_full_range_harm_v1/model_card.md).

### Preceding Selected-Tail Diagnosis

**29 October: the remaining selected-risk error is mainly underestimated harm.**
I replayed all144 frozen six-policy views. In the protected branch's selected
cohort, correcting reference error alone would still allow1,573 of1,589 known
easy-risk violations. The model's bounded adjustment also cannot reach many
observed harm values. These are diagnostic counterfactuals, not legal inputs or
proof that a wider head will work. No deployment or independent evidence changed.

[Diagnostic results](outputs/publication_readiness_2026_09/european_selected_tail_audit_v1/conclusions.md),
[interpretation and next controlled test](outputs/publication_readiness_2026_09/european_selected_tail_audit_v1/failure_analysis.md).

### Preceding Protected-Harm Experiment

**29 October: protecting benefit predictions did not make intervention safe.**
I tested three matched continuations from the same utility-trained anchors:
continue utility learning, jointly learn risk, or freeze benefit/reference
predictions and train a separate harm branch. All 54 new fits completed, followed
by the full 144-view recording readout. An earlier numerical backward failure
was repaired and preserved; none of its 37 completed fits was reused.

The protected branch kept its benefit/reference outputs exactly unchanged, but
lost to joint training at the same intervention count: **-0.276 percentage
points** in conservative utility, with a nominal 95% CI of **[-0.491, -0.077]**.
It violated known easy-risk limits in **47/72 views** and achieved complete
policy support in only **3/72**. It does not advance or change deployment.

The important distinction is between estimating harm over a fixed cohort and
estimating harm in the cases the model actually selects. The former improved
descriptively; the latter still substantially underestimates harm. I am retaining
that negative result rather than treating a protected architecture as proof of
safety. These are exposed TRAIN-internal diagnostics, not independent test
results, trajectory FDE improvements, or a new world-dynamics capability.

[Complete results](outputs/publication_readiness_2026_09/european_isolated_harm_fp64_v1/readout/conclusions.md),
[failure analysis](outputs/publication_readiness_2026_09/european_isolated_harm_fp64_v1/failure_analysis.md),
[method](outputs/publication_readiness_2026_09/european_isolated_harm_fp64_v1/method.md),
[numerical repair](outputs/publication_readiness_2026_09/european_isolated_harm_fp64_v1/numerical_note.md),
[experimental gates](outputs/publication_readiness_2026_09/european_isolated_harm_fp64_v1/gates.md).

### Preceding Decoder-Risk Factorial

**29 October: better cost prediction still does not make switching safe.**
I completed a fixed four-way experiment separating output sharing from the
training risk penalty: 72 new cost-head fits, 144,000 updates, and a complete
144-view fit/held-recording readout. All 18 replay controls reproduced their
previous training exactly. Training took 9 minutes and evaluation 111 seconds.

Removing the risk penalty improved cost prediction substantially, but increased
known easy-risk failures from **40 to 61 of 72 views** in the shared model.
Source-specific output layers improved training fit without a stable gain on
held recordings. The source-specific utility model reduced held signed-score
MSE from **4.014 to 1.952** relative to local controls, yet its same-count
conservative utility fell by **0.569 percentage points** (nominal 95% CI
**[-1.079, -0.110]**), and it failed easy risk in **54/72 views**.

This isolates a real mismatch between average prediction accuracy and safe
selection. Neither registered repair advances. I have not changed deployment,
relaxed the 2% risk budget, or selected a favorable subset. These remain exposed
TRAIN-internal diagnostics, not independent test results or FDE improvements.
The next question is how to preserve accurate gain prediction while learning
rare selected harm without distorting the same representation.

[Complete factorial results](outputs/publication_readiness_2026_09/european_decoder_risk_factorial_v1/readout/conclusions.md),
[failure analysis](outputs/publication_readiness_2026_09/european_decoder_risk_factorial_v1/failure_analysis.md),
[manuscript addendum](outputs/publication_readiness_2026_09/european_decoder_risk_factorial_v1/manuscript_addendum.md).

### Preceding Shared-Risk Study

**29 October: shared risk learning did not fix recording transfer.**
I have completed 108 new cost-head fits: 72 local models, 18 shared models with
global risk constraints, and 18 shared models with source-specific constraints.
Every model received 2,000 updates; per-source sampling histories match exactly.
All final weights were frozen before the complete 72-view recording readout.
Compared with local heads, shared-global learning loses **2.095 percentage points**
of conservative paired utility (nominal 95% CI **[-3.480, -0.825]**); sharing with
source-specific risk constraints loses **2.604 [-4.256, -1.128]**. Both also lose
at matched intervention counts and have worse signed-score prediction error.
These are utility-bound contrasts, not FDE improvements.

Source-specific constraints reduce known-label easy-risk violations from46/72
to40/72, but only1/72 views has full completion support. That is insufficient:
fewer switches and fewer reported violations do not establish a safe model.
Both repairs fail and neither advances. A complete frozen-model census shows
two problems: local models have only1/72 known easy-risk failures on fit recordings
but46/72 on held recordings, while shared models already fit their training costs
worse. Their mean fit signed-score MSE is about1.97 versus0.56 for local heads.
Pooling alone is not the repair; source-specific cost structure and the training
objective need controlled isolation.

This is a cost-learning experiment, not a new neural trajectory model. It uses
only original TRAIN packets and refits preprocessing on inner-fit recordings.
The3,000-draw bootstrap is locality-based and nominal, not independent confirmation.
Independent evaluation and deployment remain unchanged.
[Complete study](outputs/publication_readiness_2026_09/european_shared_train_risk_v1/readout/conclusions.md).
[Fit versus transfer](outputs/publication_readiness_2026_09/european_shared_train_risk_v1/fit_census/conclusions.md),
[failure analysis](outputs/publication_readiness_2026_09/european_shared_train_risk_v1/failure_analysis.md),
[manuscript addendum](outputs/publication_readiness_2026_09/european_shared_train_risk_v1/manuscript_addendum.md).

### Preceding Constraint Study

**29 October: better TRAIN constraints do not yet produce a safe policy.**
I completed a matched constraint-scaling experiment: 72 new scaled cost heads
and 72 freshly rerun controls, each trained for 2,000 updates. Every control
exactly reproduces its previous model, optimizer state and sampling history.
The forecast models, inputs, hard decision rule and 2% risk budget stay fixed.
This isolates a cost-learning repair; it is not a new trajectory predictor.

The repair works on one narrow optimization measure. Positive soft easy-risk
excess falls from **61/72 to 12/72 fixed TRAIN monitors**. Yet known easy-risk
violations on development recordings remain **58/72**. None of the 72 full-policy
views passes the finite-support completion check, so the model is **not promoted**.
The 2% budget is positive selected easy harm/reference error, not net easy ADE.

Against the unscaled control, the normalized paired lower-utility contrast is
**-1.8910 percentage points [nominal 95% CI: -3.1577, -0.7822]**; at the same
intervention count it is **-0.4845 [-0.8279, -0.1423]**. The original forest
comparison is positive, but does not rescue failed stronger controls. These are
conservative decision-utility bounds, not FDE improvement or t+50 results.

A full training-set check now makes the gap clearer. I re-evaluated all 288
frozen heads, without retraining or opening an independent test set. The scaled
model has only **2/72 known-label easy-risk violations on TRAIN**, but still
**58/72 on development recordings**. Missing outcomes remain important: only
2/72 TRAIN views satisfy the full completion check. The next question is how to
make harm prediction generalize across recordings, not simply how to increase
the penalty or tune a threshold.

I also found an error in the evaluation design. The original comparison policy
never intervenes in 29/72 views. Requiring a finite selected-risk ratio after
matching its zero intervention count is impossible. Future experiments will
separate full-policy safety from same-count utility comparisons, without changing
the 2% safety budget or dropping empty views. The old results remain failed:
correcting this design error does not repair the model's actual risk violations.

Training took 8 minutes and fixed-final evaluation 3 minutes 44 seconds on
native arm64 CPU. All 72 views across 12 localities are complete. The 3,000-draw
bootstrap resamples localities, not overlapping rows. Three cost-head seeds are
not three independently trained forecast models. Independent evaluation remains
unopened, deployment is unchanged, and the work is not yet submission-ready.

- [Complete scaling experiment](outputs/publication_readiness_2026_09/european_scaled_risk_v1/readout/conclusions.md)
- [What the repair did and did not fix](outputs/publication_readiness_2026_09/european_scaled_risk_v1/failure_analysis.md)
- [TRAIN aggregation evidence](outputs/publication_readiness_2026_09/european_risk_aggregation_v1/conclusions.md)
- [Full TRAIN census and generalization gap](outputs/publication_readiness_2026_09/european_train_census_v1/conclusions.md)
- [Prospective evaluation repair](outputs/publication_readiness_2026_09/european_gate_contract_v1/conclusions.md)
- [Latest manuscript addendum](outputs/publication_readiness_2026_09/european_scaled_risk_v1/manuscript_addendum.md)

- [Complete decision-risk comparison](outputs/publication_readiness_2026_09/european_task_risk_v1/readout/conclusions.md)
- [Constraint and selected-harm failure analysis](outputs/publication_readiness_2026_09/european_task_risk_v1/failure_analysis.md)
- [English manuscript addendum](outputs/publication_readiness_2026_09/european_task_risk_v1/manuscript_addendum.md)
- [Earlier loss comparison, including negative results](outputs/publication_readiness_2026_09/european_easy_harm_deviance_v1/readout_verified_v3/conclusions.md)
- [English manuscript, revision 5](outputs/publication_readiness_2026_09/evidence_manuscript_v5/README.md)
- [Earlier temporal and TRAIN diagnosis](outputs/publication_readiness_2026_09/evidence_manuscript_v4/manuscript.md)
- [Local and CREATE reproduction guide](outputs/publication_readiness_2026_09/local_create_runbook_zh.md)
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
