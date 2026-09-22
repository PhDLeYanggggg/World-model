# Longer Cost-Head Fitting Improves Protected Development Gain

Completed 2026-09-22. Registration `621a6eb2` preceded the real pilot and
training. All configurations, sampling, forecasts and policy thresholds stayed
fixed. The registered primary comparison passes; independent confirmation and
deployment do not follow.

## Material Passport and Evidence Status

Research-development experiment on four already explored SDD physical sites,
33 recordings and 175,756 past-eligible queries. Eight observed/twelve predicted
sampled annotation steps, stride12, annotation-pixel ADE/FDE. This is not the
historical raw-frame t50 endpoint, metric prediction or seconds-level evidence.

- `fresh_run`:12 new width128 training paths and12 width64 continuations;
  36 new endpoints,252,000 new updates and64,512,000 new draws. The wide3000
  endpoint is a prefix of its12000 path, not an independent fit.
- `cached_verified`:12 narrow3000 reference endpoints, fixed upstream EqMotion
  predictions, pair-excluded supervision and earlier cost controls.942 source
  bindings; no incomplete-supervision sample was drawn.
- `not_run`:new trajectory-predictor training, independent calibration, final
  confirmation, deployment or any new CREATE job. No closed role is opened.

Narrow/wide heads have22,978/45,954 parameters. Recorded new head fitting is
247.81 seconds, excluding loading/readout/verification, inherited narrow3000
fitting and the earlier7.75 hours of upstream producers. Narrow continuations
inherit36,000 updates/9,216,000 draws in total; these are not counted again.
This is real Torch head training, not a NumPy substitute or full world-model fit.

## Registered Primary Result

The fixed wide12000 strict-stop rule gives **3.72892% ADE improvement over CV**,
conditional site-bootstrap CI **[2.29303%,5.39095%]**, FDE improvement4.03089%
and hard ADE improvement3.74386%. Aggregate positive-easy ADE improves0.81395%
(degradation is therefore-0.81395%). Every seed improves CV and passes the
registered aggregate easy criterion. Complete exact-zero-CV harms are0.

Against the registered refitted fraction-strict reference, ADE gain increases
**2.11962 percentage points**, paired site interval **[0.51680,4.53486]**.
All four site-mean differences are positive. All four registered empirical
checks pass. This is a development result, not a population safety certificate,
new architecture contribution, independent generalization or submission readiness.

| Seed | ADE gain (%) | Easy degradation (%) | Selected | Unknown ADE | Incomplete future | Complete exact-CV harms |
|---|---:|---:|---:|---:|---:|---:|
|17|4.07794|-0.72140|10,095|138|1,405|0|
|29|3.40308|-1.01317|8,948|136|1,207|0|
|43|3.70574|-0.70727|9,522|124|1,305|0|

There are28,565 selected query/seed instances,5.41755% of527,268, including
398 unknown-ADE and3,917 incomplete-future outcomes. Unknown ADE is included in
the incomplete count, not an additional disjoint category. These are repeated
instances, not independent people. The seed mean averages errors, not forecasts.

## Fixed Factorial

All gains are native ADE versus CV, averaged equally over physical sites.
Negative easy degradation means improvement. All policies below were frozen;
neither a favorable secondary nor a seed is selected after readout.

| Width / updates | Strict ADE gain | Strict hard gain | Strict easy degradation | Matched-count ADE gain | Matched-count easy degradation | Net ADE gain | Net easy degradation |
|---|---:|---:|---:|---:|---:|---:|---:|
|64 /3000 cached|1.09819|1.49420|0.54512|5.38996|7.49659|12.17432|25.09532|
|64 /12000 continued|3.17864|3.26442|-0.58188|5.53454|6.34282|12.03087|22.02434|
|128 /3000 prefix|1.32868|1.74975|0.57578|5.42675|7.37632|12.20651|24.77997|
|128 /12000 endpoint|3.72892|3.74386|-0.81395|5.55001|6.16006|11.99134|21.52299|

The matched-count anchor remains42,422 original frozen fraction-strict choices.
All matched-count arms fail easy preservation. Every net arm harms21 complete
exact-CV query/seed instances; neither policy is promoted as protected success.

| Fixed strict-policy contrast | Difference (pp) | Conditional paired site CI |
|---|---:|---|
|Width128 minus64,3000 updates|0.23050|[0.07128,0.48565]|
|Width128 minus64,12000 updates|0.55028|[0.38262,0.71793]|
|12000 minus3000, width64|2.08046|[1.21688,2.94404]|
|12000 minus3000, width128|2.40024|[1.47237,3.32810]|
|Width-by-duration interaction|0.31978|[0.13849,0.56535]|
|Primary wide-long minus old refitted fraction|2.11962|[0.51680,4.53486]|

Training duration has the larger observed effect within this fixed matrix.
It does not prove convergence, that unlimited training helps, or that all losses
have been compared at the larger budget. In particular, the primary fraction
reference still has width64/3000 updates. A claim that the intermediate loss
itself is superior requires capacity-and-budget-matched native/fraction controls.

## Remaining Safety and Mechanism Failures

| Site | Primary ADE gain (%) | Difference from fraction (pp) | Easy degradation (%) |
|---|---:|---:|---:|
|coupa|4.01533|0.21780|-7.32530|
|deathCircle|6.31429|5.64629|2.94724|
|gates|1.96512|1.20059|1.47390|
|hyang|2.62094|1.41379|-0.35163|

**Aggregate preservation does not imply per-scene preservation.** deathCircle
easy degradation is3.34206%,2.52719%,2.97245% for seeds17/29/43. All exceed2%.
The unchanged primary protocol aggregates sites within each seed, but these
failures prevent a broad easy-safety or deployment claim. Full-grid gain lower
bounds in gates remain negative for every seed (-.07474,-.01012,-.03113pixels);
incomplete labels cannot be certified safe by observed-subset performance.

All12 selected complete populations still underestimate mean harm, ranging from
roughly2.1 to3.8 times predicted harm. These regression outputs are not calibrated
failure probabilities. Better overall gain has not solved conditional risk.

The1056 registered fitting/held-source diagnostic records show reduced fitting
native-cost MSE in every outer site. Averaged over seeds, narrow-short to
wide-long values are38.08 to26.26(coupa),19.73 to16.56(deathCircle),41.38 to28.54
(gates),47.02 to26.78(hyang). Fitting is optimistic and cannot demonstrate transfer.
For deathCircle held-source, native MSE drops183.54 to122.50, but fraction MSE
worsens.11950 to.12568 and global net-rank correlation falls.34146 to.30372.
Accuracy of one cost scale is not uniformly better ranking or calibrated safety.

The additional960 fixed-population records are explicitly **post-readout
diagnostics**. Width128 long-only strict groups have positive observed complete
net gains in all12 views. In coupa they restore2,339/2,255/2,231 complete rows,
with mean gain4.94/5.28/4.98pixels. On exactly those rows, short-head harm
predictions.78/.84/.86 become.23/.23/.20, while realized harm is.39/.43/.45.
Overcaution is reduced, but it overshoots into underestimation.

Against fraction, deathCircle's new-only complete group has3,156 repeated rows
with47.17pixel mean gain. Its205 positive-easy rows instead have-1.41pixel mean
gain and2.28pixel harm, predicted as only.14. This locates a remaining failure;
it is not permission to add an outcome-defined easy veto at inference.

Matched-count duration-wide gain is only+.12326pp, CI[-.10417,.35806], while
strict gain rises2.40024pp. Thus the improvement is primarily evident through
the fixed cost gate admitting different/more opportunities; general ranking
superiority from longer fitting is not established. This is not a calibrated
policy and its mechanism is not fully explained by a single aggregate metric.

## Verification and Statistical Scope

All48 endpoints replay2,109,072 score rows. Separate arithmetic checks144
policy/view choices,1,152 scene reductions and1,581,804 repeated supervision
rows, including tail statistics, exact-zero outcomes, bootstrap and partial-label
bounds. Shared preprocessing remains shared; fitting diagnostics replay through
the runner, not a second implementation. Separate code by the same agent is not
an independent research replication.50 scoped tests pass; the repository-wide
suite was not rerun because previously identified unrelated tests mutate shared
data-lake/report state. No full-suite success is claimed.

Intervals use3,000 resamples of **four physical sites**, after seed averaging,
not3,000 independent sites or175,756 independent windows. The sites are already
design-exposed. Secondary factorial intervals are descriptive and unadjusted
for multiplicity; only the preregistered primary determines this experiment's
primary outcome. Population protection, independent calibration and final
confirmation remain unproven.
[Eleven-part statistical interpretation audit](statistical_scope.md).

Analysis SHA256:`089b589f17010f89eedbeb0b3c68f79a615d2eb4ebae4b1ada07f9a5ace8c027`.
Post-readout audit SHA256:`c1963df53086da6e3b6e41a71597f82c785710076d6f024254f64c4043b8468d`.

## Next Evidence

Freeze this positive development result without expanding the model claim.
Next compare existing native and fraction objectives at the same width128 and
12000 updates, using the identical forecasts, causal inputs, initial states,
draws and policies. That is a fixed two-control completion, not a fresh exponent
or threshold sweep. Keep all current negative slices. Conditional harm
calibration and genuinely independent scene confirmation remain separate
requirements; original closed roles cannot be silently reassigned.

No new deployable policy, metric/seconds, true3D, foundation-model or
submission-ready claim. Stage5C and SMC remain unexecuted. The research goal
remains active and unmet.
