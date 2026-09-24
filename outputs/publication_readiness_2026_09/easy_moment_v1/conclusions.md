# Conditional Easy-Moment Experiment: Completed Negative Result

2026-09-24. Result source: `fresh_run` for 36 new moment forests and the fixed
policy evaluation; `cached_verified` for the frozen forecasts, original neural
cost heads and source lineage. No new forecasting model was trained. External
prediction, independent calibration and confirmation are `not_run` in this study.

## What was tested

The hypothesis was that predicting easy-weighted positive harm directly would
retain more useful intervention than a generic harm gate or a product of easy
probability and unconditional harm. Four exposed SDD sites, three seeds and three
fixed predictors were used. Both new rules use the same four-output forest and
exactly matched source draws. All 36 fits reached 128 trees before outcome readout.
There was no threshold sweep, outer-result model selection or predictor refit.

The primary descriptive metric is equal-site relative available-point ADE gain
over constant velocity, using 8 observed and 12 predicted annotation steps.
Coordinates are annotation pixels. These are not historical t+50 scores, seconds,
metric prediction, independent confirmation, true 3D or foundation-model results.

## Main result

| Frozen predictor / gate | ADE gain % | Hard gain % | Worst site/seed easy degradation % | Intervention % |
|---|---:|---:|---:|---:|
| Damping005 / old strict | 3.6347 | 4.7232 | 2.4944 | 12.2520 |
| Damping005 / direct joint moment | 0.1241 | 0.0004 | 0.0000 | 1.4592 |
| Damping005 / product of marginals | 0.2118 | 0.0007 | 0.0000 | 2.0453 |
| Transformer / old strict | 2.4368 | 2.2918 | 1.0669 | 6.6755 |
| Transformer / direct joint moment | 0.0134 | 0.00005 | 0.0000 | 0.2321 |
| Transformer / product of marginals | 0.0266 | 0.0004 | 0.0000 | 0.4988 |
| EqMotion / old strict | 1.6093 | 0.5271 | 0.4522 | 3.9265 |
| EqMotion / direct joint moment | 0.0013 | 0.0000 | 0.0000 | 0.0099 |
| EqMotion / product of marginals | 0.0390 | -0.00002 | 0.0000 | 0.3107 |

The direct rule's ADE CI95 is [0.0232, 0.2850]% for damping,
[0.0023, 0.0314]% for Transformer, and [0.0005, 0.0024]% for EqMotion.
These tiny positive gains over CV are not evidence that the proposed mechanism
beats the controls. Direct minus product is negative for all three actions:

| Predictor | Direct minus product, pp | Paired CI95, pp |
|---|---:|---|
| Damping005 | -0.08777 | [-0.20954, -0.01849] |
| Transformer | -0.01315 | [-0.02102, -0.00751] |
| EqMotion | -0.03768 | [-0.08268, -0.00660] |

Equal-intervention contrasts also favor product: -0.03832, -0.00743 and
-0.00111 pp respectively, with all three nominal intervals below zero.
Thus the utility failure is not solely an intervention-count difference.
All intervals use 3,000 paired physical-site bootstrap resamples, only four
development-exposed sites, and no multiple-comparison or population claim.

## Failure diagnosis

1. **Near-total refusal, not failed execution.** The direct rule keeps 7,694 of
   141,800 damping net-positive interventions, 1,224 of 286,649 Transformer
   interventions, and 52 of 364,667 EqMotion interventions. Counts sum query/seed
   instances, not independent agents. All source budgets, decisions and errors
   replay, so a failed fit or inference mismatch does not explain the result.
2. **Useful hard interventions are removed.** Of complete-label, beneficial hard
   cases admitted by the net-positive rule, the direct rule retains 53/39,404,
   113/66,527 and 0/76,806, respectively. These outcome-defined counts diagnose
   rejection after evaluation; they were never policy inputs.
3. **The zero-denominator safeguard is not the cause.** No original net-positive
   intervention has a predicted zero easy denominator. The nonzero conditional
   ratio exceeds the fixed pointwise budget in the rejected cases.
4. **Multiplying marginals is not equivalent, but correcting the algebra does
   not improve decisions.** Direct predicted joint harm exceeds the product in
   134,598/141,800, 283,283/286,649 and 363,166/364,667 net-supported instances.
   Equal-count comparisons still favor product. This is descriptive model-output
   dependence, not a universal claim about true conditional covariance.
5. **Likely mechanism, not yet a proven remedy.** A positive-harm constraint at
   every feature row is substantially stronger than controlling net easy ADE
   across a population. Smoothed estimates can assign nonzero easy harm even in
   regions where useful hard interventions dominate. More correct target algebra
   alone does not supply calibrated estimates, adequate support or an efficient
   allocation rule. This experiment does not isolate tree smoothing from
   conditional uncertainty or complete-label selection bias.

## Safety and missing outcomes

All 175,756 past-eligible windows remain indexed: 143,918 complete futures,
29,039 partial futures and 2,799 with none. The direct rule causes no observed
complete zero-CV harm and no positive-easy degradation at any site/seed. This
comes with negligible hard gain and is not a risk certificate.

Across three seeds, direct Transformer intervention includes six no-future
query/seed instances and 88 incomplete instances. Its full-grid gain lower bound
is slightly negative in at least one gates-site seed. Zero empirical easy harm
on observed labels therefore cannot establish safety on missing outcomes.
Per-site/seed metrics, p95/p99 tails and all bounds remain in analysis.json and
site_seed_results.csv. Positive-harm ratios, distinct from net easy degradation,
are in conditional_reliability.md. No unsupported labels are imputed as success.

## Verification and cost

- All 36 fresh forests completed the registered 128-tree budget; no early stop.
- Summed fitting-loop time: 876.812 seconds, not total pipeline wall time.
- Final four-target fitting MSE spans 0.05119 to 0.06648. This is not forecast ADE
  or validation loss and was not used for selection.
- Fresh-process inference/aggregate replay is exact.
- Separate arithmetic checks all 36 fit budgets, 216 decision arrays, 36 matched
  pairs, 1,728 scene reductions and 27 paired contrasts, including full-grid bounds.
- 56 scoped tests pass. The unrelated legacy suite was not rerun for this change.
- All required training/evaluation processes finished; checkpoints remain local.

Analysis SHA256: `51f3a8fb50b21ec1c02e7aaee37baf42ebd4e52310d05573ca7f3c75f064800c`.

## Decision and next experiment

**Reject this fixed pointwise easy-moment rule as a utility improvement.** Do not
replace any deployed policy or announce a neural world-model contribution. The
old protected Transformer remains a useful development comparator, not an
independently certified deployment recommendation. Historical Stage37 scores
are not directly comparable to this repaired native8/12 protocol.

The shortest next method test is a pre-registered comparison of pointwise versus
aggregate risk allocation with the same forecasts and estimated moments, followed
by source-excluded reliability checks. Preserve the existing easy tolerance and
report full intervention/coverage consequences; do not sweep this readout for a
winning threshold. Independent admissible calibration sites remain necessary
before any safety claim, irrespective of a development gain. HT21 remains
quarantined, DUT stays diagnostic, and DroneCrowd confirmation remains closed.
Joint-agent predictive contribution, robust external evidence and submission
readiness are still unestablished. Stage5C and SMC remain off.
