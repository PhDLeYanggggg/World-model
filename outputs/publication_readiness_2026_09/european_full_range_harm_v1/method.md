# A Controlled Range Repair, Not a New World Dynamics Claim

The frozen policy predicts benefit, harm, baseline reference error, easy
reference error and easy harm from causal features. The original isolated branch
can move harm by bounded odds multipliers. Near-zero anchor odds can remain
near-zero even at the largest allowed shift. A144-view diagnostic found
substantial realized selected harm beyond that branch's reachable range.

The new branch replaces this mapping, not the data, loss, base network or
training budget. A tanh output provides a fraction of the legal harm interval;
clipping keeps all-harm nonnegative and below envelope minus fixed benefit.
A second shift covers the easy-harm fraction between zero and one. Increment
form preserves the initialized anchor. An independent harm encoder is the same
size as before, while the benefit/reference base is frozen bit-for-bit.

This addresses a representational restriction, but does not solve missing
information, scarce event support or policy-conditional calibration. Its
hard clipping also creates flat gradients at interval boundaries; wider range
can make optimization worse. The repair must be judged against the frozen
isolated control at identical source/query draws, not against a shorter or
weaker fit. Both reachability and gradient geometry change, so neither can be
identified as the sole mechanism from this experiment.

No additional architecture search, loss-weight tuning, threshold selection or
early stopping is allowed. All18 finals are evaluated, across three head seeds,
along with all six controls. Same-count utility, completion-bounded risk and
empty-policy behavior remain mandatory. No local success upgrades deployment.

All data are already exposed original TRAIN records. The novelty question is
still unresolved: protected branch design or model-family combination alone
does not establish a contribution. This study tests a failure mechanism, not
independent forecasting superiority, physical safety or foundation capability.
