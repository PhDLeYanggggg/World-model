# Why the capped norm-balancing repair is not promoted

## Evidence source

Fresh run:18 new full Torch cost heads and 72 complete inference groups.
Cached verified:36 frozen controls and the earlier label-aware bank oracle.
Separate-process inference replay:all groups and all 144 control checks exact.
SOURCE fitting/resubstitution only, not independent confirmation or whole-pipeline OOF.

## Hypothesis and what actually changed

The preceding 19 exact instrumented training replays found frequent auxiliary
activation but a tiny parameter-gradient contribution. The auxiliary is positive
on 33,771/36,000 original admitted-head steps. Its all-snapshot median norm is
0.003632% of the anchor norm. Missing activation is therefore not supported.

The single registered repair uses a detached scale capped at 1e6, targeting
equal pre-clipping gradient norms. No new feature, candidate, decoder, tolerance,
data role, training duration or outcome-selected hyperparameter is introduced.
This is a simplified norm-ratio intervention, not a full GradNorm reproduction.

## Mechanism check

The new trajectories contain 28,464 zero-scale and 6,163 capped steps out of
36,000. At 163/738 monitored snapshots the scale is positive. On that conditional
subset the median achieved auxiliary/anchor norm ratio is 0.132318, p90 is 1.
The rule can exert substantial pressure when active, but cannot be described
as exact parity at every step. Dynamic admissions change with the new model;
the old all-snapshot and new positive-only summaries are different populations.
Neither is an estimate of the actual AdamW update direction or calibrated risk.

## Outcome taxonomy

|Finding|Observation|Interpretation and boundary|
|---|---|---|
|Utility loss|Balanced-minus-unit -1.516394pp, CI[-2.869731,-0.491213]|The registered primary is worse; all three seed means are negative.|
|Heavy pruning|Selected row views 265,671 to 25,828|A 90.28% intervention reduction explains a narrower operating regime, not selective-learning success.|
|Incomplete risk support|201/444 supported recording views;31 known easy violations;72 undefined easy views|Fewer violations cannot substitute for all-required-view safety.|
|Hard utility loss|Pooled known hard ADE gain 1.343824% to 0.296045%|The repair also removes useful hard-case interventions. This is a supplementary pooled statistic, not the locality-macro primary.|
|Residual actual neural use|Same balanced head +0.030673pp completion-lower, CI[+0.003886,+0.079485]|A small secondary effect remains; it cannot rescue the failed repair or establish safe deployment.|
|Regret rises|Shared-bank known regret 0.384395 to 0.400085|Avoiding switches increases missed candidate opportunity.|

## Remaining failures are not all observed harm

Of 444 overlapping recording/context/seed views, the all completion-upper risk
condition fails on171 and the easy condition on238; easy degradation upper
fails on76. All selected support is undefined on47 and easy support on72.
Conditions overlap and must not be added into one failure count.

On94 views, the all-channel completion upper fails despite known risk being
at most2%; the analogous easy count is135. These are evaluation-only arithmetic
witnesses. They show that known-only calibration cannot establish the original
complete-policy contract under unresolved outcomes. They do not identify an
inference-time subgroup or authorize imputation, cohort deletion or a looser gate.

## What this does and does not reject

It rejects this registered capped norm-ratio repair as an effective safe policy.
It does not reject all gradient balancing, prove feature insufficiency, prove
that a larger model is needed, or prove that the candidate bank lacks opportunity.
The cached label-aware whole-record bank oracle is positive and feasible in
148/148 seed-deduplicated context/record views. That is a label-informed upper
opportunity diagnostic, not deployable prediction and not a proof of causal
simultaneous-frame feasibility.

The next useful distinction is joint current-frame allocation versus independent
selection under the same candidates and risk rule, first testing feasibility.
Repeating a weight sweep or relabeling these fitting results as confirmation
would not resolve the present mechanism or evidence gap.

## Decision

Fitting gates3/6; advancement false. Independent confirmation, deployment and
submission readiness remain false. No Stage5C or SMC. Supplied detector-silver
top-down pixel/raw-frame8-observed/12-requested forecasting is not metric,
seconds-level, true3D, human gold or a foundation world-model result.
