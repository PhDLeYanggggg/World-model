# Matched Active-Bank Failure Analysis

## What Was Actually Tested

Fresh_run: 54 real Torch cost-head fits, all 2,000 updates each, followed by
72 fixed policy readouts. Cached_verified: frozen trajectory producers,
original SOURCE inputs and original cost-head controls. Only the seventh
candidate changed; six physical actions, floor, fitting rows, sampler,
preprocessing, seeds and decoder stayed fixed. This was not a new trajectory
training run, a hard-classification selector or a threshold sweep.

The question was whether the unchanged gain/harm learner could exploit the
actual active producer's new history-conditioned proposal. Its answer is
negative for the complete policy. No result from the independent roles or
the 89 controller-held recordings was scored.

## Main Findings

| Quantity | Result | Interpretation |
|---|---:|---|
| Physical-only known ADE gain vs floor | 1.946059% | Frozen comparator, not a chosen winner |
| Old-neural known ADE gain | 1.968657% | Old policy reproduced exactly |
| New-neural known ADE gain | 1.893446% | New policy is not better |
| New minus old, known | -0.075211pp, CI [-0.171084,+0.012147] | No stable positive replacement effect |
| New minus old, completion lower | -0.681429pp, CI [-1.441316,-0.218155] | Conservative evidence does not support improvement; not observed damage |
| Same new head, neural-enabled minus neural-disabled | +0.165906pp, CI [+0.054680,+0.334448] | Small actual neural-use benefit on known labels |
| Same-head neural-use completion lower | +0.125674pp, CI [-0.005768,+0.302693] | Not stably positive with missing-outcome uncertainty |
| New policy recording-risk support | 37/444 required views | Fails unchanged full-policy safety |
| New policy known easy-risk violations | 313/444 views | Missing labels alone do not explain failure |

The three head-seed replacement increments are -0.090073, -0.050546 and
-0.085014pp. A lucky average seed does not explain the negative replacement
result. Context/seed averaging precedes the 3,000-locality bootstrap; these
are conditional fitting intervals, not independent confirmation.

## Failure Taxonomy

### 1. Raw Proposal Gain Does Not Transfer Automatically To Routing Gain

The preceding active-producer history-minus-disabled known ADE contrast was
+1.160404pp, CI [-0.355089,+2.540277]. That proposal effect was itself not
stable. Replacing a candidate also changes the supervised action-cost surface
and shared head's physical decisions; it need not improve a selective policy.
The matched experiment controls that replacement rather than comparing a
different backbone or a post-hoc winning producer seed.

### 2. Shared-Head Decisions And Actual Neural Use Have Different Effects

The new head selects the neural action in 88,379 repeated row views, including
940 wholly unknown ones. Neural-enabled versus neural-disabled decoding of
the **same frozen head** has a positive known-label increment. But the full
new head is worse than the old head and is not stably better than the
separately trained physical-only head. A useful local neural action and a
poor shared policy can coexist. Neither result should be substituted for
the primary comparison. The same-head restriction is not a fourth fit.

### 3. Point-Moment Regression Is Not A Recording-Risk Certificate

The five output moments are benefit, positive harm, reference error, easy
reference error and easy positive harm. All are continuous targets. The
decoder applies the fixed 2% rule to predicted ratios, not observed outcomes.
The new head's known-selected harm is underpredicted in 52,914/263,581
repeated views (about 20.08%); easy harm is underpredicted in 42,602
(about 16.16%). Its selected reference-error bias averages +0.259655 frozen
cost-scale units, while benefit bias averages +0.043124. Inflated reference
predictions can make predicted risk ratios optimistic. These are observed
residual diagnostics, not proof of the unique causal failure mechanism.

Overall mean harm bias is nearly zero (-0.000113). This does not bound the
harmful tail of the decisions selected by the model. Lower training loss or
low average residuals are therefore insufficient safety evidence. The
[machine diagnostics](moment_diagnostics.json) preserve exact weighted
support, all five moments and their MSEs; no probability-calibration claim
is made.

### 4. Average Easy Gain Does Not Discharge The Safety Requirement

The supplementary pooled known easy ADE gain remains positive: physical
4.170441%, old 4.414932%, new 3.721732%. Nevertheless, the full new policy
violates the known selected-easy positive-harm rule on 313 required views.
Net average gain can cancel harmed cases; the approved safety rule does not.
Those are different quantities and denominators. Undefined easy support is
also retained, not counted as a pass.

### 5. Action Headroom Exists, But It Is Future-Informed

All three frozen banks have positive feasible oracle policies on 148
seed-deduplicated context/record views. This rules out a simple claim that
there is no feasible action opportunity **on these fitting labels**. It does
not show that past inputs can identify those actions or that the learned
policy is safe. The oracle reads future labels and never enters inference.
The common union-oracle regret is slightly worse for the new head (0.394587
cost-scale units) than the old (0.393820), not a new eight-action policy.

### 6. Missing Outcomes Prevent Stronger Accuracy Claims

There are 149,510 unique fitting queries across 74 recordings and 12
localities. Of these, 146,347 have at least one supplied future label and
3,163 have none. Some known queries are only partially labeled. The 897,060
context/seed row views are not independent queries. A required locality has
no requested final labels, so full-roster FDE is not_estimable. The six empty
same-count groups per contrast also keep their complete aggregate undefined.
No missing label, scene or empty control is removed to obtain a result.

### 7. This Is Not A New Independent Test

SOURCE has historical development exposure. Producer exclusions, cache
hashes and exact replay establish controlled arithmetic and lineage; they
do not restore scientific independence. Independent selection, calibration,
confirmation and DroneCrowd outcomes remain closed. No full scene-joint
decision model was trained in this experiment.

## Next Discriminating Repair

The next repair should target the mismatch between point-moment regression
and the deployed risk ratio: learn conservative harm/reference bounds or a
recording-aware constrained objective, with explicit abstention when support
is insufficient. Hold the old/new bank, original SOURCE fitting roles,
sampling, seeds, update budget and 2% evaluation rule fixed. Register the
single changed factor before any new readout. Check both actual neural use
and physical-choice changes. Do not replace the result with another raw ADE
producer run or tune a threshold on these 72 scores.

This is the next experiment, not a completed repair or a reason to open the
independent data. Deployment and submission readiness remain false.

SOURCE is supplied detector-silver bbox-midpoint pixels, obs8/pred12,
stride12 raw frames. Not historical SDD t50, verified live sensor-as-of,
human gold, metric/seconds, true3D or a foundation model. Stage5C and SMC
remain disabled.
