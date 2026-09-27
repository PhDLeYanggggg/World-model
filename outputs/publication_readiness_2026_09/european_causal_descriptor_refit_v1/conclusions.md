# Causal Descriptors Improve Mean Error, Not Safe Selection

## What Was Actually Run

This is a completed matched training experiment: 108 new risk heads, each with
2,000 updates, against 108 hash-verified signed-excess controls. The nine
forecasting banks, protected-damping floor and utility models were not retrained.
Six continuous descriptors of past motion, neighbor availability and prediction
disagreement entered a zero-initialized branch. The new models have 25,028
parameters, 384 more than the controls.

Shared initial weights, initial predictions, source-balanced training draws,
optimizer settings, update count, original-feature preprocessing and the 2%
risk screens remain matched. Added-feature scaling uses fitting sources only.
There was no held-source threshold or checkpoint search. Registration commit
`fe3a39f6` preceded training; decision freeze `a432f1a5` preceded readout.

## What Improved

- ADE gain over the protected floor: **0.2594% [0.1632%, 0.3687%]**, compared
  with 0.1848% for the control. The paired advantage is
  **0.0749% [0.0309%, 0.1382%]**.
- All 108 training monitors declined and all added branches learned nonzero
  weights. No unknown-label row was sampled for supervised loss.
- The paired held-source signed-score MSE change is
  **-0.002588 [-0.006177, -0.000148]**, a small exploratory improvement.
  It is not a calibrated risk guarantee or a separately preregistered primary.
- Easy net error remains preserved: worst view +0.3716% relative to constant
  velocity; no recorded harm to zero-error CV cases.

## What Failed

Intervention grows from **6.7818% to 8.0235%**. At the same intervention count
within each current recording/frame, the old control gains 0.2731% over the
floor. The new selector is therefore **worse by 0.0139%
[-0.0298%, -0.0022%]** against that count-matched control. Only four of the
twelve locality contrasts are positive, seven are negative and one is zero.
The apparent overall improvement must not be described as better ordering.

The primary positive-harm reduction is **undefined**, because ten dependent
held views have no defined selected-risk denominator. They are retained, not
dropped or assigned zero risk. A further **82 of 216 views exceed 2%**. The
worst defined ratio is 43.734%, based on only one known selected row. That
small denominator explains instability, but does not exempt the policy from
the registered gate. Unknown-label interventions have no assessed harm outcome.

Easy net preservation and positive-harm control answer different questions:
net gains on some cases can conceal damage to others. The joint screen fails.
**No deployment change is supported.**

## Interpretation and Next Experiment

This result rejects the simple feature-augmentation repair at this budget.
It does not show that causal motion features are universally useless. Their
small held-score MSE improvement fails to translate into the required
conditional ordering and risk control. Extra capacity is also a confound for
any claim about the six descriptors' individual meaning.

The next discriminating step is to decompose positive benefit and positive
harm for the predictions exchanged with the same-query control. This will
separate bad utility ranking from insufficient harm estimation, including
sparse support and unknown outcomes. Any subsequent repair needs a new frozen
gain/harm training contract, not a threshold sweep on these outcomes.

## Scope

All confidence intervals use 3,000 bootstrap draws over twelve opened
development-locality means, after averaging dependent producer/fit/seed views.
The three seeds belong to the forecasting banks, not three independently
collected datasets. These intervals are exploratory, not confirmation or
safety certificates. Repeated windows are not independent observations.

Data are image-local detector silver, obs8/pred12 with raw-frame stride12.
No metric, seconds, human-gold, physical-safety, true3D or foundation claim.
Independent selection/calibration/confirmation remain closed. Stage5C and SMC
remain disabled. Historical t+50 scores are not the comparator for this study.

See [results](results.md), [paired failure analysis](failure_analysis.md),
[model/data card](model_data_card.md), and [verification](verification.json).
