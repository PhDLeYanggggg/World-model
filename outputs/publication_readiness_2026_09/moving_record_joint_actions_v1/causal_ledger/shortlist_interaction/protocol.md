# Interaction Context For Fixed-Shortlist Choice

Test whether current-query agent interactions improve the final choice
within the unchanged family-diverse eight-action shortlist. Use the
already-exposed Moving A development cohort, eight observed and twelve
future annotation steps, native image coordinates. No reserved outcomes.

Keep original direct choices, forecast paths, candidate slots, ridge
objectives and weights, fitting roles (7 fit / 6 calibration / 1 held),
forecaster seeds, and opening rule unchanged. Compare the verified
unaugmented risk-aware choice to two fresh fits:

1. Observed interaction: same-query nearest distance, closest distance
   over history, recent closing, closing of the current nearest agent,
   heading agreement and soft density. Distances are normalized by the
   mean of the two existing causal motion scales.
2. Observed interaction plus rollout: early/late baseline proximity
   and change in worst and time-averaged nearest-neighbor proximity
   when changing this one agent's path and holding all others at their
   frozen causal floors. This is an image-space proxy, not collision
   prediction, a learned target, or a physical safety certificate.

Both use only current-query observations and causal forecasts as
inference inputs. Hold sources out of head fitting and calibration.
Freeze candidate choices before scoring. At most one added opening
before the first direct intervention; leave every direct action intact.
Report calibrated policy outcomes and matched query/count contrasts
separately. No identity fallback certificate, threshold change, or
evaluation reclassification. The 2% observed all/easy and signed easy
requirements and support rules remain in force.

Primary success requires more complete observed passes than the
unaugmented calibrated policy, no additional signed easy violations,
and a strictly positive paired 95% completion-lower interval from
3,000 source resamples averaging three forecaster seeds within source.
Also report first-two-query useful choice recall among the 18 original
zero-direct views. This and the matched-switch contrast are diagnostic,
not deployment or independent confirmation. No forecast retraining,
Stage5C, or SMC.
