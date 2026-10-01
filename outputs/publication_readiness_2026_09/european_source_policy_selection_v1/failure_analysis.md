# Why Source-Validation Utility Selection Did Not Repair the Method

## Completed Contrast

The fixed three-head source-validation choice is distinct from the previous
global-MSE checkpoint comparison. It does not tune transfer thresholds, fit new
networks or weaken the 2% selected-reference all/easy risk screens. Its 69
fallback choices remain in evaluation rather than being dropped.

The only three source choices passing all criteria retain a step-zero head on
eu-locality-124 in the single0/controller2 context, one per seed. MSE and initial
predictions/actions are identical there; the fixed tie rule chooses MSE. No
trained final-step head passes, and no learned improvement is demonstrated.

## Source-Validation Rejection Accounting

There are 216 candidate evaluations (72 sources x three candidates), not 216
independent source populations. Rejection reasons overlap:

- 175 select at least one row with an unknown validation outcome.
- 142 exceed easy positive-harm risk; 116 exceed all-case risk.
- 41 have no positive validation net utility.
- 20 have undefined selected reference; nine fail easy ADE preservation.
- Six pass, but these are the duplicated MSE/initial choices from the same three
  step-zero fits. Only three unique source choices survive.

Of the 41 candidates with no selected unknown outcome, 15 still fail an actual
risk screen. Therefore missing labels are not the only failure. Conversely,
41 candidates fail only because selected validation outcomes are unknown; this
identifies a concrete evidence gap, not permission to ignore those outcomes.
The two counts of 41 refer to different groups and must not be conflated.

## Transfer Failure

Only nine directional views intervene. Their observed whole-population easy ADE
does not worsen, but two exceed 2% easy selected positive-harm risk. The largest
ratio is 3.04230%, and 53 selected outcome occurrences remain unknown. A point
screen on one locality cannot be treated as an independent cross-locality safety
bound. The other 207 undefined selected-reference views do not become passes.

Identical matched actions explain the zero primary contrast. Any comparison of
the independent policies also changes coverage (16.77% to 0.264%). The latter
cannot be described as a learned ranking improvement or a world-dynamics gain.

## Next Most Informative Work

Do not launch another architecture or transfer-threshold sweep. There are two
separate obstacles: known-outcome conditional risk error, and unsupported
selected outcomes. First determine whether the 41 unknown-only rejected
source-validation candidates can be bounded using their existing causal rollout
disagreement, retaining the same selected-reference risk denominator and budget.
Existing Euclidean disagreement bounds should be reused and checked against the
native partial-label estimand, not assumed to cover it.

Any such bound must place worst-case harm on unknown outcomes and must not treat
them as zero or remove them using future-validity gating. It is a finite-sample
missing-outcome accounting bound, not cross-source statistical calibration.
Register a new comparison before using it to select policies. If the bound is
uninformative, preserve that result and quantify the label-support gap. Improving
unknown accounting alone cannot repair the known-outcome risk failures.

The main research goal still needs a genuinely useful learned intervention rule,
source-held statistical calibration, strong same-protocol controls and independent
confirmation. These results narrow the failure mechanism but do not satisfy that
goal or CVPR submission readiness. No deployment promotion or scientific-unit
upgrade. Stage5C execution and SMC stay off.
