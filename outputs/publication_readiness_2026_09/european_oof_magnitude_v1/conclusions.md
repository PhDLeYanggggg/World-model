# Magnitude Readout: Completed, No Promotion

## What Was Tested

The repaired auxiliary can rank harmful events more accurately, but that
does not ensure good expected-cost estimates. I tested an honest fitting-only
OOF magnitude readout, with exactly the same two-slope estimator for true
auxiliary, shuffled auxiliary and strong cost-only controls. All predictions
were frozen before this source-held readout. This is not a trajectory trial.

Fresh computation completed 1,008 native Torch nuisance/auxiliary heads,
2,016,000 optimizer updates, 432 closed-form readouts, 144 outer views and
1,728 direct MSE checks. Cached controls and frozen outer heads were reused,
not presented as new training. No unknown-label rows were sampled as zeros.

## Main Finding

The fixed scientific gates fail. No forecasting model or deployment policy
is promoted. Primary is expected easy-harm cost MSE on positive envelopes.
The following counts are six assignment-level descriptive bootstrap intervals,
not six independent trials or one pooled trajectory-improvement estimate.

| Full-input comparison | Positive / negative / overlapping-zero CIs | Point-estimate range |
|---|---|---|
| Scaled true auxiliary vs scaled cost-only | 2 / 0 / 4 | -2.91% to +1.04% |
| Scaled true auxiliary vs raw true auxiliary | 2 / 0 / 4 | +1.20% to +27.89% |
| Scaled true auxiliary vs scaled shuffled | 2 / 0 / 4 | -3.20% to +1.13% |
| Scaled cost-only vs raw cost-only | 1 / 0 / 5 | -1.77% to +26.21% |
| Scaled shuffled vs raw shuffled | 2 / 0 / 4 | -2.48% to +27.04% |

The two positive matched-control comparisons are producer0/controller1
(+0.128%, 95% CI +0.055% to +0.250%) and producer1/controller0
(+1.040%, CI +0.075% to +2.737%). Both also beat matched shuffled labels.
The four other intervals cross zero, including negative point estimates.
They must not be discarded or relabelled as successful generalization.

Motion-only scaled true vs scaled cost-only is 1 positive, 1 negative and
4 overlapping; vs scaled shuffled it is 1 positive, 2 negative and 3
overlapping. Although scaled-vs-raw MSE has 4 positive motion-only intervals
in each of the three arms, that common effect is not auxiliary-specific.
Full/motion families also differ in forecasts and event populations, so this
is not a matched feature-removal ablation.

## Why It Does Not Advance

Scaling true auxiliary worsens positive-envelope coverage-log error versus
its raw control in 4/6 full-input intervals. One top10 harm-capture interval
is negative. The analogous cost-only and shuffled controls also have 4/6
and 5/6 negative coverage-log intervals. Lower squared error alone does not
satisfy the registered harm/coverage guards. Coverage here is an expected
harm-mass diagnostic, not empirical coverage of a predictive interval.

True auxiliary still improves positive-envelope harm-presence AUROC over
scaled cost-only in 5/6 intervals and over scaled shuffled in 5/6. That
narrow ranking signal remains useful research evidence, but it neither
passes the magnitude gate nor establishes calibrated deployment safety.

## Limits and Next Question

The fit commonly shrinks both harm moments. Full-input median easy-harm
slopes are 0.327 (cost-only), 0.318 (true) and 0.315 (shuffled); no slope
hits its 0 or 8 bound. Easy-membership disagreement between inner-producer
and outer-producer cuts ranges from 0.137% to 15.012%, median 2.525% across
144 dependent views. Training-size and cut-definition transport are plausible
limitations, not proven causes. The next controlled diagnostic should isolate
them before another auxiliary architecture or threshold sweep.

All results are historically exposed source development. Seeds are averaged
inside locality; 3,000 paired resamples use only four held localities per
assignment. Intervals are descriptive, unadjusted and assignments overlap.
Independent selection, reserved calibration and confirmation remain unopened.

See `results.md` for every registered interval, `failure_analysis.md` for
the causal limits, and `project_gap.md` for the next discriminating question.
Actual replay/test completion is recorded separately in `verification.json`.

Obs8/pred12 native annotation steps; detector pixels. No metric, seconds,
human-gold, true3D, foundation or physical-safety claim. Stage5C and SMC off.
