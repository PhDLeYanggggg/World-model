# Probability Damage and Compensating Cost Errors

## Material Passport

Interpretation addendum to the sealed fitting-only diagnostic, not a new model,
policy or held result. Arithmetic seal:
`cbf8f36e29dc697a1a465f9b88153584a179be07b07716f8643a4b00c511897c`.
This addendum is written after that seal and is not an additional independently
tested efficacy result. The source results are [analysis.json](analysis.json)
and [results.md](results.md).

CREATE37579489 completed all108 groups;37579750 repeated them exactly. Runner
times were191.83s and283.23s; scheduler times3m26s and4m54s. The latter includes
startup/teardown. Independent arithmetic15,552checks and30 scoped tests pass.
All216 heads were already trained; this diagnostic performs zero updates.

## What the Fitting Data Show

The source/query-balanced marginal objective rises from0.00278057177 to
0.00283937155, a2.1147% relative increase. Only42/108 risk-priority heads improve.
This uses every known fitting row, not the128-query training-monitor subset;
the earlier46/108 monitor count is a different scope, not overwritten.

The average factor-replacement contributions to that loss increase are:

| Predicted component replaced | Objective change | Loss-increasing groups |
|---|---:|---:|
| Easy occurrence probability | +0.000311531 | 86/108 |
| Conditional reference cost | -0.000007346 | 6/108 |
| Conditional positive harm | -0.000245385 | 41/108 |

The contributions average all six replacement orders and sum to+0.000058800.
They describe arithmetic in the frozen predictions. They do not identify the
causal effect of retraining just one branch or justify deploying a favorable
mixture selected from these fitting scores.

Occurrence Brier worsens0.111875 to0.160719. Conditional reference MSE also
worsens0.003240 to0.018265, and conditional harm MSE worsens0.005187 to0.008145.
Thus the negative reference/harm replacement contributions do NOT mean those
conditional predictors became more accurate. Their product with occurrence can
partly compensate other errors. The occurrence-harm doubled cross term becomes
more negative (-0.001024 to-0.004508), confirming cancellation in the fixed
telescoping identity. These terms are order-dependent, not unique causal shares.

Easy rows with realized positive harm contribute79.81% of uncapped and84.23%
of risk-priority row-risk MSE under the fixed global weights. That partition's
contribution increases; non-easy and easy-zero-harm contributions decrease.
The main remaining fitting error is not established to be a lack of zero-harm
examples. This does not contradict older experiments with different models,
targets and weighting, where zero-target rows dominated.

## Decisions and Next Controlled Test

1. Retain the failed-risk verdict and current protected deployment. Do not sweep
   the auxiliary cap, tune held thresholds or pick one of the eight mixtures.
2. The next mechanistic hypothesis is whether separating/fixing occurrence
   estimation prevents risk/cost learning from degrading it. Compare a single
   specified separation against a matched continued-training control, with the
   same fitting queries and source exclusions; register before training.
3. Check usefulness as well as risk. Separately completed fitting-only signed
   gain labels now support benefit-ranking diagnostics. Their construction did
   not evaluate ranking. The current component diagnostic remains not_run for
   that question because its original packets did not contain signed benefit.

The next test is not yet trained or validated. Neither probability repair nor
fitting-loss improvement alone can repair the structural unsupported views,
prove independent-source calibration, or satisfy the selected-risk primary.
Do not change the denominator, discard abstaining views or open independent
selection/calibration/confirmation to make the screen pass.

## Claim Boundary

All108 groups reuse only the existing source-training development localities;
216 source views are not independent samples. No new held outcomes or actions
were computed. Observation8/prediction12, raw stride12, image-local detector
silver. No metric, seconds, human-gold, physical-safety, true3D, foundation or
submission-ready claim. Stage5C/SMC remain disabled. The core method still needs
a positive compliant risk-control result and independent confirmation.
