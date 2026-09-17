# Single-Factor Diagnostic Repair: Context Pooling

The first registered fit-fold probe completed 24 fits, all worse than the
training-only prior in Brier score. There are 365 windows but only 31 agents and
45 stopped runs. Stationary-window annotation-change rates differ markedly:
59/81 in ETH and 129/284 in Hotel. Ordered geometry/motion has 83 features.

This adaptive follow-up tests one repair before adding a neural start head:
replace ordered neighbor slots with 13 permutation-invariant summary features
(six for geometry only). Retain the same fit-only rows, physical-scene folds,
labels, seeds, model families, hyperparameters and prior. No additional tuning,
balancing, loss change or development labels. Report every setting, including
failure and the empty Zara stationary fold.

Current nearest/mean/max distance and observed counts describe geometry. Mean
past path, speed, closing, closest approach and straightness describe motion.
All are deterministic summaries of the already sealed causal feature vectors.
No new data or targets enter the feature computation. This is a post-diagnosis
training-internal experiment, not independent confirmation of a selected method.

Reduced dimension may mitigate small-sample fitting, but a positive classification
result would still not predict direction or prove trajectory improvement. If
cross-scene probability error stays worse than the prior, do not launch a new
stationary-start neural correction on this evidence. Preserve negative results.
