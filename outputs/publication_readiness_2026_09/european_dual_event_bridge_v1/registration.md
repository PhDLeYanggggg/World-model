# Dual-Event Policy Bridge: Registered Development Experiment

## Material Passport
Mode: experiment execution. Source: prior frozen source models and opened
model-selection data. This record fixes a new learned comparison before fitting.
No human participant study, independent calibration, confirmation or deployment.

## Question and Motivating Failure
The preceding six-locality readout found easy degradation as large as 8.216907%
under all-event control. Easy-event add-only policies all preserved easy means,
but were not uniformly better than training-selected motion baselines. All and
easy policies use different fallback forecasts: their old risk scores cannot
be conjoined as if they described the same decision.

Learn whether to bridge from the fixed old_stop easy policy R to the fixed
old_stop all policy P. Both entire policies, including the motion floors and
forecaster, were fitted on producer A's four localities only. New heads fit
on disjoint controller B's four localities. No B-trained add_only endpoint is
used to make B supervision. All six ordered A/B pairs and seeds 17/29/43 are
included, with no best-seed or best-pair reporting.

## Matched Intervention and Labels
383 causal inputs: the established geometry/relative-rollout cost features,
CV rollout and four frozen policy bits. Future targets are labels only.
Utility predicts positive ADE benefit and harm for P versus R. Two separate
risk heads predict reference ADE and positive harm moments for (a) all labelled
rows and (b) CV-positive easy rows, using A's frozen training-only easy cutoff.
Unknown outcomes remain NaN and are never sampled for fitting. The maximum
R/P rollout separation bounds cost labels without reading future masks.

Controls use the same pair and trained utility: utility only, all risk only,
easy risk only, both risks. Each predicted constraint has the unchanged 0.02
ratio budget relative to R. Identical rollouts and latest-step stops cannot
trigger learned intervention. Report R, P and a ridge dual-risk control too.
Learned constraints are NOT conformal calibration or a safety theorem.
Inherited reference harm and CV-defined easy degradation remain separately
audited; improving R alone is not sufficient evidence of success.

## Fixed Training and Readout
54 real neural heads, each 2,000 updates, batch 256, width 64, AdamW 0.0003,
plus 54 ridge fits at alpha 0.01. Single-process native arm64 CPU4/interop1;
checkpoint/heartbeat every 200 updates. The first 100 updates of one utility
head are a runtime/resume pilot inside its 2,000-update budget. No tuning sweep.
Use source-balanced training and training-only normalizers. Preserve checkpoints
on failure and require 10 GiB free disk. CREATE is only needed if local pilot
cost/resources justify it; read-only queue access does not mean a job was run.

Readout reuses the six already-opened model-selection localities and all 38,102
past-indexed rows, including unknown and partial future labels. These are not
new independent evidence. Freeze all decisions before fresh error calculation.
Report all 18 dependent groups, all seven policies, old easy add_only and all
add_only controls, six motion baselines, raw neural, and the producer-training
selected motion baseline. Primary: equal-locality ADE gain over old easy
add_only, not merely over R; easy worst-locality degradation at most 2% and no
new zero-CV harm. Also hard, complete-future, endpoint, tail, intervention and
per-locality predicted/realized all/easy risk. Bootstrap 3,000 locality resamples
and report three-seed averaged paired comparisons. Intervals are conditional
development summaries, not multiplicity-adjusted confirmatory tests.

No automatic candidate promotion or threshold adaptation. A negative result is
retained. Twelve calibration and six confirmation localities stay closed.
No additional data role or external dataset admitted. Eight observed/twelve
requested native annotation steps, detector-track image pixels, no metric,
seconds, physical safety, human-gold, true-3D or foundation claims. Stage5C and
SMC remain off.

## Literature Boundary
Selective regression can improve pooled performance while worsening a group;
this is an existing concern, not a new claim of M3W. Here the protected event
is outcome-defined CV-easy and the alternative is another forecast, not
demographic-group abstention. See [Shah et al., ICML 2022](https://proceedings.mlr.press/v162/shah22a.html).
[Learn then Test](https://arxiv.org/abs/2110.01052) provides a multiple-risk testing
route when valid calibration conditions hold. Predicted moment constraints
alone are not that procedure. This experiment does not open calibration data
or claim a distribution-free guarantee.
