# Next Discriminating Repair and Remaining Research Gap

This turn made experimental progress: 432 exact training reconstructions and
2,160 new intermediate observations. It did not improve a deployable model.
The overall research goal remains active, not complete or submission-ready.

## Next Minimal Intervention

Preregister a fitting-prior initialization repair before training. Change only
the auxiliary logit intercept from inherited easy-membership prevalence to
the cap-event prevalence calculated with the existing fitting-only weights.
Keep cost outputs, shared weights, features, sampler, loss coefficient,
optimizer, seeds and 2000-update budget fixed. Retain true and locality-shuffled
event arms and the exact no-auxiliary control. Reuse the verified originals;
do not rerun forecasts or select intermediate checkpoints.

The hypothesis is narrower than generic loss reweighting: removing the
wrong-prior transient might preserve cost learning while retaining task
information. It is motivated by post-hoc metadata, not yet established.
Require fixed-budget actual training and preregistered cost/guard comparisons.
If it only lowers classification BCE, it fails the cost-repair objective.
Do not open independent roles or tune exposed test thresholds to rescue it.

## Still Required for a Main Method Claim

1. Reliable expected gain/harm learning under matched strong cost controls.
2. Scene-joint decisions versus independent-agent and scene-uniform choices
   at matched intervention/risk budgets, including easy and worst-scene harm.
3. Independent calibration and final confirmation after the viable method is
   frozen, with adequate independent-scene support. Source replay is not a
   substitute for these evidence roles.
4. Strong matched public baselines, justified statistical claims, reproducible
   scientific contribution, complete English manuscript and Chinese tutorial.

The existing projection and severity negatives are retained. Do not assume
that more layers, longer training or a large sweep will resolve the failure.
This initialization test is a bounded falsification step, not a promise of
success or a new architectural contribution on its own.

Local native arm64 training was stable; CREATE was checked read-only and no
remote task was altered. No new paid resources were used. Pixel/native-step
only, no physical-safety or true3D/foundation claim. Stage5C and SMC remain off.
