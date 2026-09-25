# Conditional Risk Failure on New Localities

This is post-readout diagnosis, not a new selection rule. No checkpoint, threshold,
split, mask or deployment action was changed after seeing these outcomes.

## Where the Failure Comes From

The worst add-only easy case is producer0/seed17/all-risk/controller1 on locality125.
It contains775 positive-CV easy rows. CV mean ADE is2.786323 image pixels. The old
controller has2.665212 ADE (+4.346640% gain versus CV); add-only has3.015273 ADE
(-8.216907% gain versus CV). The protected motion floor has2.757900 ADE and therefore
does not explain this particular failure by itself.

There are283 new interventions on these easy rows.146 are beneficial and137 harmful.
The harmful additions sum to327.019295 pixel-ADE units, while useful additions save
only55.722017. Net damage is271.297278, or12.563547 percentage points of the easy CV
error sum. This changes an existing4.346640% improvement into8.216907% degradation.
Counting more beneficial than harmful actions is consequently misleading: magnitude
matters. Mean predicted added harm is0.756467 versus realized1.155545 on these actions.

The whole-locality all-risk ratio for this view is1.461670% relative to the old
controller's all-event error, within2%, while positive harm on the easy additions
is15.832153% of the old controller's easy error. These are different denominators
and events. Passing an all-population constraint does not protect the conditional
easy population. The predicted all-event ratio is only0.190715%, also showing
underestimation; it is not a calibrated certificate.

## Target and Baseline Differences

All seven add-only easy failures use the all-risk target. The18 easy-risk views
retain nonnegative mean gains over CV in every locality and have no new exact-zero-CV
harm. Their108 repeated locality/view positive-harm ratios relative to the old
controller have maximum0.006042 and no value above0.02. These are observations on
six model-selection localities, not108 independent safety tests.

However, the conservative easy-target branch remains below the source-training-
selected motion baseline in some views (worst-1.773321% all-ADE). The old controller
also fails easy in six all-target views. Neither adding nor removing interventions
can assume that an inherited controller is already safe under scene shift.

Ridge reaches larger improvements in some views but also has seven negative all-ADE
intervals and worst easy degradation6.024246%. Raw neural predictions have easy
degradation24.774235% to30.675096% and harm exact-zero-CV rows in every view. Their
best hard gains cannot justify unrestricted deployment.

## What Is and Is Not Established

- Supported: small all-ADE gains from frozen add-only policies transfer to these
  six new model-selection localities. No new feature/threshold fitting was needed.
- Supported diagnosis: all-risk preservation and easy-risk preservation are not
  interchangeable, and small populations of large harmful switches dominate failure.
- Not established: a policy that uniformly beats the strong baseline and preserves
  easy cases across all configurations, calibrated risk, final confirmation, or
  scene-joint contribution. The preceding joint-control negative result remains.
- Not an excuse: future-label missingness and detector-derived tracks are limitations,
  but they do not erase the measured easy failures on the common supported labels.

The next experiment should improve the candidate/floor and the event-conditional
gain/harm objective on training data, using these now-opened localities only for
declared model selection. Do not tune on calibration or confirmation outcomes.
