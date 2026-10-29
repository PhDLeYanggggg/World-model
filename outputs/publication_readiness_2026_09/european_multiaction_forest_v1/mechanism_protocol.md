# Fitting-Only Cross-Estimator Calibration Check

Specified while primary training is running, before its first readout. This
does not change models, targets, thresholds, primary outcomes or the frozen
training registration. It is a descriptive mechanism check, not a new gate.

After the fixed readout, replay all 54 forests and their 54 static-scene neural
comparators on original fitting packets only. All 1,332 recording/context/head/
bank views remain. For both estimators measure the same five known target sums
on: all legal bank actions, the frozen neural choice region, and the frozen
forest choice region. Both estimators must share identical observed target
mass within each matched region. Choice hashes must reproduce the readout.

Report predicted/observed H, EH, R and ER ratios with median, finite count and
undefined count. Medians condition on a positive observed denominator, not on
passing safety. Unknown selected costs remain unknown and are counted, not
imputed. All-legal repeated candidate rows are not a deployment policy. Exact
separate replay and a lossless aggregate archive are required.

This separates a changed estimated moment in a fixed region from changed
selection composition. It cannot uniquely attribute a difference to estimator
capacity: the ordinary forest and neural objectives/decoders differ. Neither a
better fitting ratio nor less MSE establishes held-role safety, independent
calibration, causal identification of the failure mechanism, or deployment.

No new optimization, inner-held tuning, independent role, future-input feature,
Stage5C or SMC is permitted by this diagnosis.
