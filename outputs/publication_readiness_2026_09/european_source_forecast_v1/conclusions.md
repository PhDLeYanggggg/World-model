# European Squares Source Forecast: Completed, Not Deployable

## What Ran

`fresh_run`: eighteen real Torch Transformer fits, seeds 17/29/43, three frozen
source-locality folds and two producer roles. Each fit reaches exactly 4,000
updates; total 72,000. Each model has 88,514 parameters. The first 100 updates
were a real training pilot resumed inside, not added to, the fixed budget.
No early stopping or checkpoint selection on held outcomes was performed.

The registered cohort contains 318,969 complete-past targets from 163 recordings
and twelve opened source-training localities. Of these, 311,922 have some future
labels and 7,047 have none. Missing labels are masked, not an inference feature
or a reason to remove a target. The 8-observed/12-requested task uses raw stride
12 and image pixels. This is not the historical raw-t50 experiment, a full raw
dataset pass, verified online sensor data, meters, seconds, true 3D or a
foundation model. No RGB, JEPA or image-grounded world-model contribution is
tested by this motion/neighborhood Transformer alone.

## Positive Signal and Its Boundary

Mean-seed ADE gain over the baseline chosen using only the other fitting
localities is **4.1135%**, with a 3,000-resample conditional locality-bootstrap
interval of **[1.3672%, 6.9485%]**. This averages seed errors, not predictions.
Nine of twelve localities improve; the worst locality degrades by 4.1483%.
All seeds and locality results are retained in [the tables](results.md).

This comparison is not superiority over every strong fixed control. On the
same equal-locality CV-relative scale, the neural model gives +2.1576%, while
the predeclared fixed damping-0.97 baseline gives +3.9755%. The fitting-locality
baseline choice does not always generalize to the excluded locality. These
readout observations do not change that choice retrospectively.

## Failed Safety

| Seed | Positive-easy degradation vs CV | Worst easy-locality degradation | Zero-CV cases harmed |
|---|---:|---:|---:|
| 17 | 13.7319% | 75.9522% | 4 / 4 |
| 29 | 13.6479% | 75.6037% | 4 / 4 |
| 43 | 14.3936% | 81.0359% | 4 / 4 |

The unchanged positive-easy limit is 2%; allowed added harm at exactly zero
CV error is zero. Both fail. Four zero cases are scarce support, not four
independent safety trials. The training-selected fallback itself degrades
positive-easy errors by **15.4759%** against CV. A budget for harm relative to
that fallback cannot by itself protect easy cases relative to CV.

The fitting objective minimizes average masked, training-site-normalized ADE.
It does not enforce either safety condition. That objective/constraint mismatch
is verified from the implementation; it does not prove that loss weighting
alone will fix generalization. The minibatch losses are noisy and are not
evidence of convergence. See [actual loss traces](training_losses.md).

## Verification and Limits

- `cached_verified`: source/role/cache/prediction identities are checked and the
  complete published metric readout is recomputed without retraining.
- `fresh_run`: separate checkpoint inference on the first 128 excluded rows of
  every model reproduces the saved predictions exactly: 18 checks, maximum
  absolute difference zero. This is a sampled checkpoint replay, not a second
  full retraining or full-population inference replay.
- 127 relevant tests pass together. This is not the complete legacy test suite.
- Every model's excluded source fold is absent from its training draws. The
  nested producer chain used by the subsequent cost experiment also excludes
  the outer locality.
- Reserved model-selection, risk-calibration and confirmation data remain
  closed. Bootstrap localities share fitted models and are opened development
  sources; the intervals are conditional exploratory evidence, not independent
  confirmation or a calibrated safety guarantee.
- Detector-track annotations, partial-future support and site heterogeneity
  limit the claim. A bounded duplicate screen is not universal independence.

## Decision

Do not promote the model. Do not change risk limits or pick a new winner from
this readout. Stage5C and SMC remain off. The next version needs a genuinely
CV-relative safety reference, separate positive-easy/zero-event modeling and
strong fixed-baseline controls. Those changes require a new frozen protocol;
they must not be presented as if they had already passed independent testing.

Primary evidence: [analysis](analysis.json), [metric replay](verification.json),
[checkpoint replay](checkpoint_replay.json), [registration](registration.md),
[matrix](matrix.json), [execution record](../european_source_execution_20260924.md).
