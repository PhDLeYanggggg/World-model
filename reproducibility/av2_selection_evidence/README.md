# Selection and harm coverage: scalar replay

Run `python -I -B -S verify.py` from this directory. No model,
network connection, trajectory, or third-party library is needed.

The anonymized table contains six held development-cluster
totals. It reproduces the paired point estimate and enumerates
all 46,656 equally weighted bootstrap cluster draws to obtain
the exact conditional percentile interval. This independently
checks the sign of the separate 3,000-draw estimate; the
percentile endpoints need not match exactly.

The same harm estimator scores fixed and optimized choices on
the same scenarios. Three seeds are averaged within scenario
and are not independent observations. The optional-veto counts
are a separate frozen threshold replay, not proof of risk
control or independent confirmation. No identifying scene
keys, raw data, checkpoint weights, or held forecasts are
included.
