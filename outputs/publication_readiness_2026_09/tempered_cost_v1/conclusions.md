# Intermediate Weighting Repairs Tail Ranking, Not Protected Intervention

Completed 2026-09-22. Registration `5a9d02ab` preceded fitting and readout.
Only the fixed exponent-one disagreement weighting changes. No threshold,
primary comparison, feature, seed or outcome role changes after registration.

## Evidence Status

`fresh_run`:12 real Torch cost-head fits,36,000updates,9,216,000draws, fixed
readout, full score replay and separate arithmetic checks. Each head has22,978
parameters. No unknown-supervision row was drawn. Recorded head fitting totals
28.08seconds on CPU4; this excludes loading/verification and the completed
7.75hours of upstream pair-excluded EqMotion fitting. The100-update pilot was
resumed, not discarded or called a full experiment.

`cached_verified`: frozen EqMotion forecasts, nested supervision and previous
cost controls;904dependency bindings. `not_run`: new forecast-model training,
independent risk calibration, independent final confirmation or deployment.
Four source sites are design-exposed. This is not full world-model training
or an untouched final test.

## Registered Primary Result

The tempered strict rule gives **1.09819%** equal-site ADE improvement over CV,
scene-bootstrap CI **[0.56186%,1.99872%]**, and1.20020%FDE improvement. Hard ADE
improves1.49420%; positive-easy ADE degrades0.54512%in aggregate. All three seed
aggregates have positive gains and easy degradation below2%. No observed
complete exact-zero-CV outcome is harmed.

But the primary contrast against refitted fraction-strict is **-0.51112
percentage points**, CI **[-2.49011,1.24907]**. The combined primary criterion
fails. This is not promoted to a successful or deployable policy.

| Seed | ADE gain (%) | Easy degradation (%) | Selected | Unknown ADE | Incomplete future | Complete exact-CV harms |
|---|---:|---:|---:|---:|---:|---:|
|17|0.84662|0.13972|954|29|212|0|
|29|1.55309|1.12785|2,186|33|401|0|
|43|0.89485|0.36777|1,623|30|327|0|

These are repeated query/seed instances, not independent people. Total strict
selection falls20,703to4,763, or0.9033%of527,268query/seed instances.
92unknown-ADE and940incomplete-future selected outcomes remain.

**The seed-average easy gate is not a per-scene guarantee:** deathCircle/seed29
has3.20327%easy degradation. Full-grid absolute-gain lower bounds in gates are
negative for allseeds (-.06297,-.07922,-.04323annotation pixels), because
incomplete outcomes permit harm. Unknown outcomes and undefined zero-reference
percentages are not converted to safe zeros.

## All Fixed Comparisons

All gains below are native ADE relative to CV, equally averaged over sites.
Negative easy degradation means improvement. Controls are cached-verified, not
newly trained. Matched counts use the original frozen transferred fraction-strict
anchor, unchanged across heads.

| Head / fixed policy | ADE gain (%) | Hard gain (%) | Easy degradation (%) | Complete exact-CV harms |
|---|---:|---:|---:|---:|
|Refitted fraction / strict|1.60931|0.50847|-1.78803|0|
|Refitted native / strict|2.96117|4.28682|2.20284|6|
|Tempered / strict|1.09819|1.49420|0.54512|0|
|Tempered / net|12.17432|14.81292|25.09532|21|
|Refitted fraction / matched|4.07919|7.05031|6.78282|0|
|Refitted native / matched|5.26674|8.88612|7.46381|0|
|Tempered / matched|5.38996|9.05827|7.49659|0|

Matched tempered-minus-native is+0.12322pp, conditional scene CI[.06200,.19702],
with identical counts and positive mean difference in each site. This is a small
ranking improvement, not protected success. Matched tempered-minus-fraction is
+1.31077pp, CI[-.06983,3.87993]. The net policy's large gain fails easy and
exact-zero protection; it cannot replace the failed primary. No favorable seed
or secondary policy is promoted. Intervals use3,000physical-site resamples after
seed-error averaging, with four exposed sites, not independent query CIs.

## Failure Taxonomy

1. **The intended tail repair occurs.** With the prior fitting-q99 disagreement
   cuts, deathCircle held net-gain Spearman rises from.095/.148/.146for fraction
   loss to.595/.593/.601for tempered loss. Fitting values rise from
   .210/.312/.353to.626/.640/.650. Benefit/harm tail bias shrinks. This is source
   development, not independent or cross-domain confirmation.
2. **Moderate low-risk opportunities are lost.** In coupa, old fraction-only
   strict groups have2,200/1,970/2,164complete rows with realized mean net gains
   5.52/5.40/5.36pixels. New tempered-only complete groups contain55/52/87rows.
   On exactly the old-only rows, tempered harm is.96/.83/1.00versus observed
   .50/.46/.50. Many fail the unchanged strict ratio. Their causal past motion
   is much smaller than replacement groups. Tail improvement alone is not enough.
3. **Scene tradeoffs cancel.** Strict tempered-minus-fraction differences are
   coupa-3.26105pp, deathCircle+1.80121pp, gates-.17729pp, hyang-.40735pp.
   Better high-motion selection does not recover the aggregate primary gain.
4. **Ranking and reliable cost estimates remain distinct.** Nine of twelve
   tempered strict populations underestimate mean harm; hyang's three views
   overestimate it. These costs are not calibrated failure probabilities.
5. **Aggregate protection hides a failure slice.** deathCircle/seed29exceeds2%
   easy degradation, and incomplete-label bounds also permit harm. Complete-zero
   observations do not establish full-population protection.
6. **No runtime failure explains the result.** All fits finish the fixed budget
   with finite losses and exactly matched reference draws; replay and arithmetic
   verification pass. This does not prove convergence, adequate model capacity
   or sufficient causal information.

The312fitting/held records and120same-population comparisons are explicitly
post-readout diagnostics. Selection-group means use complete outcomes only;
unequal missingness is reported. They do not replace the primary analysis or
establish full-population counterfactual effects.

## Next Test

A sweep over more exponents or held-source thresholds would not answer the
remaining question. Let X include causal disagreement D and Y be paired costs.
For positive w(X), the unrestricted minimizer of
E[w(X)||f(X)-Y||^2 | X] remains E[Y|X]. These losses thus change finite-model
fitting priorities, not the ideal conditional-mean target. At D=0 the geometric
identity fixes both costs to0. This elementary observation is not a novel
theoretical contribution or a finite-network guarantee.

The next controlled test should separate finite capacity from inadequate
optimization, keeping features, draws and policies fixed. Low- and high-
disagreement fitting diagnostics both matter. Capacity/budget changes require
prospective registration, not a remedy already assumed to work. Independent
scene calibration and final evidence remain unresolved; original closed roles
must not be opened to rescue this result.

## Verification and Claims

All12heads replay527,268scores. Separate code checks1,581,804repeated fitting
cost rows,36policy/view choices and288scene reductions including uncertainty,
tails, exact-zero outcomes and incomplete-label bounds.99scoped tests pass.
Same-agent separate arithmetic is not independent research replication.
Analysis SHA256: `a332cd3d84537524e1dc0d262061b77c6734a708b623f5e19f53547ec69f4388`.

Eight observed/twelve predicted sampled annotation steps, stride12, annotation
pixels. Raw-frame t50 is supplemental. No metric/seconds, true3D, foundation,
population-safety, deployment or submission-ready claim. Stage5C/SMCremain
unexecuted. M3W's larger research goal remains unmet.
