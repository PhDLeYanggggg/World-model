# Occurrence-Severity Supervision: Completed, No Stable Safe Neural Advantage

25 September 2026. `fresh_run`: 72 Torch risk heads, 144,000 optimizer updates
and 144 policy views. Forecasts, original utility heads, source roles and the
2% risk rule are `cached_verified` and unchanged. This experiment tests risk
supervision, not a newly trained trajectory model. No deployment changes.

## What Changed

The preceding experiment bounded predicted harm by causal forecast separation
but still underestimated harm on selected rows. Here two identical networks
predict reference error mass, harm occurrence probability and conditional harm
severity. The matched control learns only the resulting error moments. The
hurdle arm additionally supervises occurrence and positive-only severity. Both
use the same inputs, initialization, sample draws and training budget.

All 72 heads were fitted and replayed before new outcome readout. Both predefined
all-event and easy-event targets are retained; neither is selected afterwards.
The [registration](registration.md) predates training in commit `7a1cd0d9`.

## Protection Improves in One Respect, Not All

| Risk arm | Neural observed safety | Neural worst positive-easy degradation | Damping observed safety |
|---|---:|---:|---:|
| Original | 9/18 | 5.5327% | 16/18 |
| Geometric envelope | 8/18 | 17.8014% | 18/18 |
| Matched product MSE | 6/18 | 17.2546% | 18/18 |
| Occurrence-severity hurdle | 6/18 | 0.6746% | 18/18 |

These are maxima and pass counts across the same 18 views per arm, not one
paired treatment-effect estimate. Positive-easy means the fixed easy subset
with positive reference error. Observed safety also requires no added error on
zero-CV-error rows. All 18 hurdle views meet the positive-easy 2% limit, but
12 harm zero-error cases. The remaining six have no zero-error cases in their
readout and therefore do not establish zero-error protection.

The [posthoc zero-reference audit](zero_reference_audit.json) finds only four
distinct such rows, all in one locality. Two fitting folds contain none of
them. Every hurdle view evaluated on those rows harms one to three, with added
ADE of 0.05294 to 0.92109 image pixels. The error is not merely numerical
epsilon, but no physical interpretation is justified. No tolerance was changed.

## Accuracy Does Not Establish the Main Claim

Against the same-architecture product-MSE control, all-event neural controllers
improve all-ADE in 8/9 comparisons, with seven strictly positive conditional
intervals. Easy-event neural controllers lose all-ADE in all nine comparisons,
with all nine intervals negative. The loss decomposition changes the
protection/coverage tradeoff rather than uniformly improving selection.

Against equally protected damping, the hurdle neural candidate has only one
positive all-ADE point estimate in 18 comparisons: +0.2451%, conditional 95% CI
[-0.1032%, +0.6369%]. No all-ADE interval is strictly positive; 15 are negative.
The comparison uses the same risk rule, not the same intervention count.

There are two positive hard-subset intervals, both under fold 2's easy-event
target: seed 29 gives +0.5874% [+0.2603%, +0.9032%], and seed 43 gives +0.3308%
[+0.0601%, +0.6326%]. These are retained, not hidden. However, 15/18 hard
intervals favor damping, neither positive result has zero-error readout support,
and the two selected intervals are not multiplicity corrected. They do not
establish a stable cross-scene neural advantage or authorize deployment.

An auxiliary posthoc check compares harm-probability Brier score with a constant
estimated on fitting rows. Neural hurdle heads improve that score in all nine
easy-event views, eight with positive conditional intervals, and six of nine
all-event views. Thus some occurrence signal is learned. A better probability
score alone is not a reliable intervention policy or a dynamics contribution.

## Evidence and Limits

- [All registered views and contrasts](results.md), [aggregate metrics](group_metrics.json).
- [Objective comparisons](objective_comparison.svg), [training moment losses](training_moment_loss.svg).
- [Failure analysis](failure_analysis.md), [gates](gates.md), [execution receipt](execution_notes.md).
- [English research addendum](paper_addendum.md), [Chinese operation guide](operation_zh.md).

All 72 checkpoints reproduce the first 4,096 excluded-index scores each, all
72 samplers match, all 144 full metric views reproduce, and 72 old/geometric
control views match. There are 218 passing tests across 34 scoped files, not
a full legacy-suite run. Engineering reproducibility is not scientific success.

Each fit uses four fitting and eight complete-producer-chain-excluded localities,
but all twelve European Squares localities are already opened development data.
Three seeds and 3,000 locality-bootstrap draws provide conditional uncertainty;
the overlapping views are not independent experiments. Reserved calibration
and confirmation remain closed. The next useful diagnostic is matched-coverage
risk ranking and support-aware protection, not relaxation of the risk rule.

Released detector tracks, image pixels, 8 observed / 12 predicted steps at raw
stride 12. Not legacy raw-frame t+50, seconds, metric, human gold, physical
safety, true 3D or a foundation model. Historical Stage37 is not recertified.
No Stage5C execution, SMC, deployment promotion or submission-ready claim.
