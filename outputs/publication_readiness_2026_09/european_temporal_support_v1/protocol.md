# Temporal Context and Event Support

## Material Passport
Source-development experiment, registered before new fits. Parent: cost-shape
v1, verified at 33e3fa4a. Forecasts and nuisance neural heads are cached-verified;
new contextual ridge fits and diagnostics are fresh_run. This is not a new
neural dynamics model, independent confirmation, or a deployment change.

## Question
Does ordered causal motion and neighborhood history add predictable information
about easy-harm cost beyond frozen risk scores and the earlier seven summaries?
Global shape/scale sweeps have stopped. There is one fixed ridge coefficient,
no threshold search, and no search on outer or independent outcomes.

## Exclusion
Retain six producer/controller assignments, three seeds, full/motion-only pairs,
and four outer source-locality exclusions. Within each outer view, hold one of
the three fitting localities for an inner information screen and fit on two.
For fitting rows in locality L, use a cached reference trained only on the other
fitting locality M. Both references exclude inner scoring and outer localities.
Score the inner locality with its already frozen two-locality cost-only head.
Both fitting and scoring targets use the two-locality fitting easy cut. The
single-locality reference's own easy cut can differ, which remains an explicit
stacking/transport limitation. Never fit a second-level head using ordinary OOF
rows whose producer has seen the scoring locality.

The new probe uses only the three fitting-locality targets. No outer-source
target scoring is performed. Source memories may map other source labels, but
the probe never accesses their rows. Independent selection, reserved risk
calibration and confirmation remain closed. Existing source-development
outcomes are not made independent by this exclusion design.

## Fixed Arms
- Score only: log1p envelope and four moments, two causal moment ratios.
- Old summary: add seven prior context summaries and missingness masks.
- Ordered history: add 68 heading-aligned, path/observed-width-normalized past
  position, displacement, acceleration, speed, turn and motion-mask features.
- History plus neighbors: additionally add 68 time-ordered neighbor support,
  relative position, distance and closing-motion summaries. At most eight
  selected, complete-history, current-visible neighbors; not the entire crowd.

All four are closed-form ridge .1 probes for the residual of expected easy
harm. Fitting-only equal-locality positive-envelope supported weights set
normalization, response RMS and coefficients. Standardized inputs clip at 8.
Inference clips easy harm to the frozen predicted all-harm cap. Other moments
stay unchanged. The cap's label-derived unavoidable MSE is diagnostic only.
Prediction accepts no target, future endpoint, easy label or measured error.

## Support and Interpretation
Report unique recordings, tracks, agent-query keys, event-bearing supports,
harm-mass concentration and greedy non-overlapping recording-query intervals.
The source adapter uses raw frame stride 12: obs8/pred12 spans query-84 through
query+144. Greedy disjoint intervals are not independent people or scenes.
Do not sum dependent views or seed replicas into sample size.

Primary stays expected easy-harm MSE on positive-envelope known rows. Report
top10 harm-mass capture, coverage-log error and all-harm MSE guards; these are
not trajectory ADE/FDE gains or easy-degradation percentages. Inner-fold
outcomes are scored only after all probe predictions are committed. Average
the three outer-context values within scoring locality, then three seeds,
then 3000 paired resamples of four localities, separately for six overlapping
assignments and both families. Intervals are exploratory and unadjusted.

A follow-on outer conditional-head study is justified only if full-input
history-plus-neighbors beats score-only and old-summary with at least two
positive and zero negative primary intervals each, without negative guard
intervals. This is a feasibility screen, not a new paper success criterion.
All negative and absent-support results remain reported. If the screen fails,
do not sweep ridge widths/thresholds; diagnose cap/support/transport and seek
better causal observations or genuinely new training scenes as warranted.

Detector pixel coordinates and native annotation-step protocol only. No
metric/seconds, physical safety, human-gold, true3D or foundation claim.
Stage5C execution and SMC remain disabled.
