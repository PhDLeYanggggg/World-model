# Predictor-Specific Costs: Protection Recovers, Primary Gain Test Fails

Completed 2026-09-22. Registration commit`1a9d26ed` preceded cost-head training
and its outcome readout. All models, budgets, thresholds, common-count controls
and the primary comparison remained fixed. No seed or secondary arm is promoted
after observing this result.

## Evidence Status

`fresh_run`:36cost heads,108,000optimizer updates,27,648,000draws, fixed readout,
complete cost-head replay, separate arithmetic verification and post-readout
same-population diagnosis. All36heads have22,978parameters and finite loss traces.
Their combined recorded fitting time is76.65seconds on CPU4, plus data/verification
overhead; this does not include7.75hours previously spent fitting18upstream
EqMotion producers. The100-update pilot was resumed, not restarted.

`cached_verified`:18newly verified pair-excluded EqMotion training producers,
12fixed outer EqMotion predictors,36frozen Transformer-trained cost heads and
841dependency bindings. No unknown-supervision row was sampled for fitting.

`not_run`: independent calibration, independent final confirmation and deployment.
The four physical SDD sites have influenced research design and remain exposed.
All intervals below are3,000physical-site bootstrap resamples after seed-error
averaging, not independent confirmation or row-level resampling.

## Primary Result

The refitted fraction-strict rule improves equal-site ADE over CV by**1.60931%**,
conditional scene CI**[0.71627%,3.03928%]**. FDE gain is1.73064%; hard ADE gain
is0.50847%. Positive-easy ADE improves1.78803%, and no observed complete exact-CV
outcome is harmed. All seeds have positive ADE gains and meet the easy limit.
The largest site/seed positive-easy degradation is0.30995%, in deathCircle/seed29.

However, the frozen transferred fraction-strict rule had3.32610%ADE gain. The
registered refit-minus-frozen difference is**-1.71680percentage points**,
CI**[-3.78525,0.31186]**. Its lower endpoint is not positive, so the combined
primary criterion fails. Improved protection is not evidence of superior overall
intervention or permission to replace the failed primary outcome.

| Seed | Refit ADE gain (%) | Positive-easy improvement (%) | Selected instances | Unknown ADE | Incomplete future | Complete exact-CV harms |
|---|---:|---:|---:|---:|---:|---:|
| 17 | 1.59517 | 1.79687 | 6,631 | 31 | 518 | 0 |
| 29 | 1.47939 | 1.66455 | 6,564 | 43 | 574 | 0 |
| 43 | 1.75335 | 1.90267 | 7,508 | 30 | 637 | 0 |

Counts include repeated query/seed instances, not independent people.
The primary rule selects20,703instances versus42,422for the frozen strict rule.
It still selects104unknown-ADE and1,729incomplete-future instances. Unknown
outcomes are not safe or zero error. Conservative full-grid absolute-gain lower
bounds are positive for every site/seed, but this bounds average error difference
on the present queried population, not subgroup safety or future deployment risk.

## Complete Fixed Policy Comparison

Positive easy degradation is worse; negative values mean improvement. All gains
are equal-site native ADE gains versus CV. Matched-count policies all use the
frozen fraction-strict count; they are offline ranking controls, not deployment.

| Fixed policy | ADE gain (%) | Easy degradation (%) | Hard gain (%) | Complete exact-CV harms |
|---|---:|---:|---:|---:|
| CV | 0.00000 | 0.00000 | 0.00000 | 0 |
| Uncontrolled EqMotion | 11.04350 | 33.36170 | 15.44117 | 34,584 |
| Past-stop EqMotion | 12.18575 | 27.63510 | 15.27848 | 21 |
| Frozen direct net | 7.33658 | 14.56474 | 9.81618 | 16 |
| Frozen direct strict | 3.18013 | 4.38976 | 4.59573 | 0 |
| Frozen direct matched | 4.59005 | 5.78483 | 6.79770 | 13 |
| Frozen bounded-native net | 10.82199 | 24.51029 | 13.82477 | 19 |
| Frozen bounded-native strict | 6.42425 | 13.78037 | 9.11995 | 6 |
| Frozen bounded-native matched | 4.80337 | 8.03300 | 8.43440 | 0 |
| Frozen fraction net | 10.95135 | 20.56543 | 13.42570 | 21 |
| Frozen fraction strict | 3.32610 | 3.78188 | 4.32725 | 0 |
| Frozen fraction matched | 4.93853 | 7.65561 | 8.52057 | 0 |
| Refit direct net | 5.11264 | 8.78984 | 6.87964 | 6 |
| Refit direct strict | 1.39586 | 1.09556 | 2.06867 | 0 |
| Refit direct matched | 4.14042 | 4.00841 | 5.72452 | 1 |
| Refit bounded-native net | 11.09830 | 21.16721 | 13.74244 | 17 |
| Refit bounded-native strict | 2.96117 | 2.20284 | 4.28682 | 6 |
| Refit bounded-native matched | 5.26674 | 7.46381 | 8.88612 | 0 |
| Refit fraction net | 11.00653 | 24.13983 | 13.31557 | 21 |
| Refit fraction strict | 1.60931 | -1.78803 | 0.50847 | 0 |
| Refit fraction matched | 4.07919 | 6.78282 | 7.05031 | 0 |

At common counts, fraction refit-minus-frozen is-0.85934pp, CI[-3.30999,0.59313].
Reducing intervention is therefore not the only unresolved issue; improved
ranking is not demonstrated. The secondary bounded-native matched contrast is
+0.46337pp, CI[0.17668,0.83441], but easy degradation remains7.46381%. It is not a
safe positive primary result. Native and fraction training losses have different
weighting and are not compared as if lower numeric loss implies better safety.

## Failure Taxonomy

1. **Support repair is real but insufficient.** The fraction of all feature
   cells exceeding10fitting standard deviations drops in every view: frozen
   values1.14-3.01%become0.050-0.331%. This combines predictor-specific fitting
   and preprocessing; it does not isolate a causal normalization effect.
2. **Conditional harm remains underestimated.** On the new rule's complete
   selections, actual/predicted mean harm is1.56-2.85in all12views. These are
   continuous costs, not calibrated failure probabilities. Comparing these
   ratios directly with the old policy would confound different selected rows.
3. **Same-population checks expose overcorrection as well.** On the old
   deathCircle/seed17 strict selection, actual harm is5.23annotation pixels,
   old predicted harm1.30and new predicted harm17.01. The new head also predicts
   only17.76benefit versus32.90observed. A simple global harm multiplier cannot
   be presumed to repair this simultaneous under/overestimation.
4. **Ranking loses high-benefit motion.** Under matched counts in deathCircle,
   the three frozen-only complete groups have mean realized net gains80.45,
   62.97and49.47pixels; the replacement groups give9.22,13.01and4.61. The discarded
   groups have larger past displacement per sampled step and forecast
   disagreement. Completeness differs, so these subset means are diagnosis,
   not full-population counterfactual guarantees. This scene contributes
   -4.47087pp to the matched fraction contrast.
5. **Safety improvement is mostly a different operating point, not a solved
   cost model.** The new strict rule is a useful exploratory protected tradeoff,
   but hard gain falls from4.32725%to0.50847%. Conditional calibration and useful
   high-benefit ranking remain missing.
6. **No implementation failure was needed to explain the result.** All heads
   complete the budget, have finite losses, replay, share matched arm draws,
   and preserve clean fitting exclusion. Training completion does not prove
   sufficient optimization or adequate features; those remain hypotheses.
7. **Independent evidence is still absent.** Four development sites and three
   seeds cannot establish cross-domain or population-wide safety. Neither an
   elementary disagreement bound nor predictor-specific refitting is, by itself,
   a new paper contribution.

The120same-population diagnostic records were computed after readout and are
explicitly exploratory. They change no model, threshold, choice, split or metric.

## Verification and Next Evidence

All36cost heads replay1,581,804new score rows. A separate implementation checks
36fitting-supervision sets,252policy/view choices and2,016scene reductions.
This is same-agent verification with shared preprocessing recomputation, not
independent research replication.91scoped tests pass. Required processes exit0.
Analysis SHA256:
`260700b79a0e0bda415fc54b4e09df66ca3c2b4c2732573ec5c32d2d7985dd9d`.

Next, test whether the high-disagreement benefit/harm ranking error is already
present within clean fitting predictions or arises under held-source/predictor
shift, without choosing a threshold on these outcomes. Use training-defined
motion/disagreement strata and source-excluded checks before registering another
change to targets or features. Independent calibration/final roles still require
their separately authorized evidence; original closed roles remain unopened.
Do not just extend training or promote the favorable secondary arm.

Eight observed/twelve predicted sampled annotation steps, SDDstride12,
annotation pixels only. No metric/seconds, true3D, foundation, new deployment,
Stage5C or SMC claim. The project remains not submission-ready.
