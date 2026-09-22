# Conditional Risk Changes Across Sites, Not Just Feature-Range Outliers

2026-09-22. **fresh_run** read-only fit/held diagnosis and exact repeated replay;
**cached_verified**24 cost checkpoints and prior decisions. No new fit or policy
change in this audit. Four explored SDD sites,8observed/12predicted annotation
steps, pixels. No independent calibration, metric/seconds, deployment, Stage5C
or SMC claim. This extends, not overwrites, the temporal intervention result.

For the ramp head, selected complete-label mean harm is underestimated in13/36
fitting-site groups versus12/12 held-source/seed groups. Realized/predicted harm
ratio median is0.8874 on fitting groups and2.7861 on held groups (range1.8054 to
3.9868). The uniform comparator gives0.9937 and2.4808. Fitting-weighted selected
harm is also underestimated in all12held groups. Aggregate-population error
alone hides the decision-region problem. In-sample fit diagnostics are optimistic
and do not establish calibration or complete convergence.

Among11,566 complete exact-zero-CV queries, only7have a nonzero last historical
displacement:5coupa and2hyang. These are overlapping queries, not7independent
events. The harmed hyang_seed17 query has all356features inside the fitting
coordinate-wise range and maximum standardized magnitude2.8911. It is not
caught by a simple range/extrapolation rule. All5moving-zero fitting queries
were sampled,197totaldraws; this is rare support, not sampler omission.

Its32nearest complete moving fitting queries contain11harmful and21beneficial
outcomes for the ramp. Mean harm is1.5562pixels versus0.8892predicted. The nearest
rows span3fitting sites, but neighboring windows are not independent. This is
descriptive local ambiguity, not proof of irreducible error or permission to
create a held-case-specific veto.

The evidence supports conditional-risk generalization as the next question.
It does not prove that domain shift is the only cause: some fitting groups also
underpredict, and representations/targets/finite support may all contribute.
Do not default to another epoch/threshold sweep or train on these held outcomes.
The next fixed comparison is a standard ExtraTrees cost regressor using the
same features, forecasts and fitting supervision. No tree result exists here.

Reproduce: `.venv-pytorch/bin/python scripts/audit_m3w_temporal_fit_support.py`.
Both executions completed and reproduced the immutable aggregate JSON exactly.
Seven dedicated unit tests cover weighted/conditional means, exact-zero moving
support, invalid weights and normalization. The full legacy suite was not run.
See [aggregate evidence](fit_support_diagnosis.json). Shared data/code checks
are engineering verification, not independent research confirmation.
