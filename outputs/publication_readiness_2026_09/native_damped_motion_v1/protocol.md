# Past-Only Speed And Turn Dynamics Under The Original Budget

Use the exposed Moving A cohort (34 recordings, 14 sources, three frozen
forecaster seeds) and the native forecast bank. CV is the original
fallback and 2% risk reference; OLS4 remains the second accuracy
reference. Keep observation 8/prediction 12, all labels and masks, and
exactly the original genuine replacement counts per query.

Anchor each new path at the current position and OLS4 velocity. Estimate
speed change from the median of the last five backward speed differences.
Bound acceleration to a quarter of the larger of OLS4 speed and median
recent speed; bound rollout speed to twice that scale. Estimate signed
turn from the last five observed consecutive displacement pairs, require
at least three same-direction changes of at least five degrees per
annotation step, and bound turn to 20 degrees per step. Set invalid pair
angles to zero. Integrate exponentially decaying increments using fixed
half-lives of 3 and 8 annotation steps. The speed-only and turning
options have identical speed schedules at each half-life. None of these
parameters is fitted or selected by outcomes. Treat new differences
within one millionth of an image pixel of CV as the unchanged path.

Two complementary evaluations:

- Hindsight: add the two speed-only options, then add the two turning
  options. Keep the native options. Reuse the original dual-reference
  maximin test and 3/12-second limits; re-score checked witnesses and
  never transfer a negative bound to a larger menu.
- Past-only: keep the recipients and counts from the frozen gain policy.
  On those recipients only, replace its path by the 3-step half-life
  speed option if absolute past speed change reaches 10% of the scale,
  then prefer the matched turn option if the observed heading condition
  holds. Do not substitute identity paths. Compare speed vs original
  and turn vs speed with the original scorer and the paired source
  bootstrap. No new threshold search, training, or safety certification.

The exact-count ceiling is diagnostic. The direct past-only comparison
is exposed development and cannot change deployment. No external
outcomes, data changes, metric or seconds claims, Stage5C or SMC.
