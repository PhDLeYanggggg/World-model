# Frozen Songdo score ranking

Run `python verify.py` to reconstruct the calibration prefix and the
fixed-budget ranked-versus-uniform comparison. This uses anonymous
scores, risk ratios and paired utility summaries from the original
cohorts. It does not train a policy, read trajectories or count an
undefined risk ratio as a pass. See `protocol.md`. The inherited
archive contains the earlier evidence and the three Songdo cases.
