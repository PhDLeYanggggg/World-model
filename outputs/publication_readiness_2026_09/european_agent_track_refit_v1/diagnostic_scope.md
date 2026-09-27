# Additional Diagnostic Scope

Fixed before comparative outcome scoring. This does not amend the registered
primary, training, seeds or forecast selection. No threshold is tuned.

1. Observation-only association response: first256 eligible held-source histories
   per model, two complete neighbors, fixed past-slot reassignment1/3/5.
   Compare output sensitivity, not accuracy or physical realism. Future labels
   are not read. The trained grouped model should distinguish these associations;
   whether that improves actual prediction is a separate main experiment.
2. Existing output envelope: after prediction freezing, use future labels only
   to compute the oracle Euclidean projection lower bound at each valid target
   step. Record zero-motion misses, unreachable target fraction and output-radius
   usage. A fixed0.95 usage statistic is descriptive, not a policy threshold.
   This isolates an output-support limitation but cannot prove that unbounding
   would improve prediction. Never feed the oracle bound back into inference.

Report every fixed producer/locality/seed and easy/hard subset. Do not select
a new main endpoint or a deployment policy from these diagnostics. No independent
selection/calibration/confirmation access, retraining, Stage5C execution or SMC.
