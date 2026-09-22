# Matched Objective Controls: Partial Gain, Primary Contribution Not Established

## Material Passport

Registered commit `06f9a635` preceded the real pilot and training. This study
adds 24 fresh native/fraction cost-head fits and retains 12 hash-verified,
already trained intermediate-loss heads. Forecasts and causal features are
cached-verified, not newly trained here. Fixed readout and full checkpoint
replay are fresh computations. All four SDD sites have informed research
design; this is developmental evidence, not independent confirmation.

The task observes eight and predicts twelve sampled annotation steps, with
SDD stride12. Errors remain annotation pixels. The main statistic is the
equal-site mean relative ADE improvement over causal constant velocity (CV),
after averaging the three seed errors. This is not raw-frame t+50, a seconds
claim, or a comparison with the historical Stage37 percentage.

## What Was Controlled

All three objectives use the same 45,954-parameter bounded head, width128,
12,000 updates, batch256, initialization seeds17/29/43, preprocessing, causal
356 features, pair-excluded EqMotion predictions, training-row draws and
decision rules. Only the cost-error weighting differs: native squared error,
error divided by forecast disagreement D, or error divided by D squared.
The intermediate objective is the fixed candidate, not a selected winner.
No thresholds, data roles, normalization or goals were changed after readout.

The 24 new fits complete 288,000 updates and 73,728,000 row draws, with zero
unknown-supervision draws. Recorded head-fit time is 245.01 seconds, excluding
I/O, verification and upstream forecasting. The cached intermediate references
account for a separate 144,000 earlier updates and 36,864,000 draws. The resumed
100-update pilot is part of the new total, not additional training.

## Fixed Strict-Policy Results

Positive gain means lower error than CV. Positive easy degradation means harm.

| Objective | ADE gain % | FDE gain % | Hard gain % | Easy degradation % | Complete exact-CV harms |
|---|---:|---:|---:|---:|---:|
|Native|3.06719|3.34135|4.30451|2.49977|0|
|Fraction|1.65767|1.78569|0.78701|-1.58864|0|
|Intermediate, fixed candidate|3.72892|4.03089|3.74386|-0.81395|0|

Candidate ADE gain remains 3.72892%, conditional site CI [2.29303,5.39095].
Its exact scores, choices and summaries equal the earlier frozen wide/long
capacity result. Reusing it does not constitute a fresh independent fit.

| Registered paired candidate contrast | ADE gain difference, percentage points | Conditional 95% site CI | Lower bound positive? |
|---|---:|---|---|
|Intermediate minus native|0.66173|[-0.27228,2.18556]|No|
|Intermediate minus fraction|2.07125|[0.66261,4.31900]|Yes|

**The joint objective-contribution check fails.** Both comparisons were required
to have positive lower bounds. The first interval crossing zero does not prove
equivalence, and the positive point estimate does not establish superiority.
The old 2.12-point comparison against an under-budget control is not used to
claim that the new loss beats both fair controls. The new control repair does
not erase the earlier capacity-duration result; it answers a different question.

## Protection Is Not Solved

Every candidate seed has positive overall CV gain and aggregate easy degradation
below2%, but deathCircle easy degrades 3.34206%,2.52719%,2.97245% across seeds.
Its seed-mean degradation is 2.94724%. Native also fails aggregate easy, and
its gates/seed43 easy degradation reaches10.01102%. Fraction has no site/seed
easy violation in this readout, but smaller gain and no independent safety
calibration. It is a useful control, not a post-hoc new deployment.

All 12 selected-complete views per objective underestimate mean harm. Actual
to predicted harm ratios span2.58-6.42(native),1.38-3.41(fraction), and
2.06-3.78(intermediate). Thus neither global cost fitting nor zero observed
complete exact-CV harms certifies the safety of interventions.

| Strict objective | Selected query/seed instances | Unknown ADE | Incomplete future |
|---|---:|---:|---:|
|Native|21,488|561|4,455|
|Fraction|22,006|211|2,386|
|Intermediate|28,565|398|3,917|

Unknown and incomplete categories overlap. They are not safe outcomes or
independent additional samples. The candidate's full-grid lower gain bound in
gates remains negative for all three seeds. Even fraction has a negative lower
bound there in seed17. Future completeness never enters the selection rule.

## Other Fixed Policies

| Objective/policy | ADE gain % | Easy degradation % | Complete exact-CV harms |
|---|---:|---:|---:|
|Native, net-stop|11.28694|21.19026|16|
|Fraction, net-stop|11.49374|21.32264|21|
|Intermediate, net-stop|11.99134|21.52299|21|
|Native, matched count|5.22158|6.97564|0|
|Fraction, matched count|4.99686|5.57510|0|
|Intermediate, matched count|5.55001|6.16006|0|

Matched count is42,422 query/seed interventions for each arm, anchored to the
original frozen fraction rule. At that common count, intermediate-minus-native
is+0.32844pp [0.22116,0.45602]; intermediate-minus-fraction is+0.55315pp
[-0.01689,1.58365]. These secondary descriptive intervals do not rescue the
failed primary comparison or the easy degradation. The larger net-stop scores
are unsafe and are not promoted.

## Evidence and Limits

`analysis.json` records all nine arm/policy summaries, scene/seed slices,
tail errors, bounds,36 selected-cost diagnostics and792 train/held fitting
diagnostics. `replay.json` verifies36 endpoint checkpoints and1,581,804 scores.
The separate arithmetic receipt and execution details are recorded alongside
this report. Separate code by the same agent is not independent replication.

The3000 paired bootstrap resamples contain four physical sites, not3000
independent scenes. Three seeds do not repair repeated research-design exposure,
missing futures or absent independent calibration. No new model is deployed.
No metric/seconds, true3D, foundation-model, submission-ready, Stage5C or SMC
claim is supported.

## Next Scientific Step

The budget confound is now resolved, but objective superiority and conditional
protection remain incomplete. The next targeted diagnosis should compare
harm underestimation and missed gain on the same causal support for all three
fixed heads, using fitting-only support definitions. That diagnosis must inform
a prospectively fixed risk-learning repair, not another threshold search on
these held outcomes. Independently assigned scene calibration and confirmation
remain necessary before a deployment or generalization claim. Original closed
roles remain unopened.
