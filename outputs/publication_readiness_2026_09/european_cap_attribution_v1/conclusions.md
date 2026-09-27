# Relaxing the Cap Does Not Repair the Context Model

## Material Passport
Fresh_run: deterministic output attribution over 432 inner fitting-locality
views, 14 predefined contrasts and 3,000 paired locality-bootstrap draws per
assignment. Cached_verified: 1,728 ridge models, their original capped outputs,
neural forecasts, nuisance heads, rows and labels. There are zero new fits or
gradient updates. Registration efa546f3 and prediction freeze dff8cf9e precede
new scoring. This is source-development analysis, not independent confirmation.

## Main Result
The frozen cap is not hiding a reliable temporal-context gain in this probe.
For full-input history plus neighbors, replacing the predicted all-harm cap
with the causal envelope produces zero positive, five negative and one
overlapping primary interval. Its point gain ranges from -11.82% to -0.20%.
The corresponding motion-only comparison is 0 positive / 4 negative / 2 overlap,
with point gains from -36.22% to -3.37%. All four input arms have zero positive
primary intervals for relaxing their cap, in both forecast families.

The matched relaxed-cap context comparisons also fail:

| History plus neighbors compared with | Full positive / negative / overlap | Motion-only positive / negative / overlap |
|---|---:|---:|
| Score only |2 /3 /1|1 /0 /5|
| Earlier seven summaries |0 /2 /4|0 /6 /0|
| Ordered history without neighbors |1 /3 /2|0 /3 /3|

The full comparison against summaries ranges from -3.24% to -0.55%; every
motion-only summary comparison is negative, with a worst point of -163.83%.
Partial gains versus a weaker score-only control do not overturn those results.

Raising all-harm to preserve the nested easy/all relation does have a secondary
signal: all-harm MSE improves in four full and five motion-only intervals
versus the frozen-cap version, with no negative intervals. The full point gains
are only 0.005% to 0.427%. This is retained rather than hidden, but it does not
repair the primary easy-harm deficit, tail guards or matched context failures.

## What the Decomposition Shows
Among 216 dependent full-input history-neighbor views, relaxing the cap lowers
easy-harm MSE in 62 and raises it in 154. Event-label error contribution improves
in 203 views, but 141 of those have worse total MSE because the zero-label
contribution offsets that benefit. The motion-only lower/higher/equal counts
are 47/168/1. These are descriptive dependent-view counts, not independent votes.

The median changed-row fraction is 12.80% for full and 10.05% for motion-only.
Without the coupled all-harm change, those rows violate the nested easy/all
prediction relation. Zero easy-harm labels include non-easy samples as well as
non-harm events; they must not be called exclusively easy-case false alarms.

The squared-error difference identity passes for all 1,728 probe vectors,
maximum absolute discrepancy 3.31e-13. The earlier large pointwise ceiling floor
was not evidence that the expected mean was biased low. This direct comparison
instead shows that removing the particular ceiling often exposes harmful
corrections. It does not prove the ceiling optimal for all estimators.

These are expected-cost MSE results, not trajectory ADE/FDE, t50 gains or easy
trajectory degradation. Four-locality intervals are exploratory and unadjusted;
six source assignments overlap. Missing motion-only guards remain missing.

## Decision
All three scientific screens fail. Do not promote a joint conditional head,
remove the deployment constraint or sweep another cap/temporal ridge setting.
The next question is the quality and information in the causal observations,
not another projection threshold. Independent selection/calibration/confirmation
remain closed. No new neural dynamics or deployment improvement is claimed.
Stage5C and SMC remain disabled; the overall research goal remains active.

See [complete contrasts](results.md), [all displayed assignments](cap_attribution.svg),
[absolute costs and counts](absolute_costs.md), [failure analysis](failure_analysis.md)
and [next boundary](project_gap.md).
