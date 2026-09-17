# v7 Supplement: Small Forecast Repair, No Joint Advantage

Both completed model families, three seeds and all twelve cost-head/policy
combinations per family were evaluated under the decision frozen before v7
development scoring. No threshold or deployment selection used these results.
Raw50 is the exact first five steps on the development recordings' ten-frame
grid, taken from the original twelve-step forecast with unchanged normalization.
It is not a separately conditioned forecast, interpolation or seconds horizon.

## Uncontrolled Prefix Results

| Seed | CV-skip raw50 ADE gain | CV-skip raw50 FDE gain | Motion-bounded raw50 ADE gain | Motion-bounded raw50 FDE gain |
| --- | ---: | ---: | ---: | ---: |
| 17 | -3.082492% | -1.193259% | +0.042166% | +0.060449% |
| 29 | -2.761527% | -1.258698% | -0.046954% | -0.006700% |
| 43 | -1.904494% | -0.287103% | +0.005591% | +0.023255% |

The bounded model does not have a consistently positive raw50 result without
selection. Its full-path-defined easy slice still degrades by 29.13%, 172.85%
and 91.58% on prefix ADE. All these negatives stay visible. The already selected
primary bounded policies have prefix ADE gains +0.001695%, +0.008054% and
+0.000313%, with easy degradation 0.786%, 1.091% and 0.134%. These are not new
raw50-selected policies or independent confirmation.

## Matched Actual Counts

The CV-skip arm selects identical agent identities under joint and independent
count-matched routing in every combination; all ADE/FDE differences are zero.
The bounded arm has nine zero combinations. The three moderate ridge cases
have joint-minus-independent primary ADE differences:

| Seed | Difference | Interpretation |
| --- | ---: | --- |
| 17 | +0.000000116083 | joint worse |
| 29 | +0.00000328304 | joint worse |
| 43 | +0.000000482345 | joint worse |

All three nonzero effects favor independent selection. The bounded matched
control changes 196 agent-query switch records across 453,300 repeated exports;
these are not independent samples. Nonzero matched scene-query counts range
176--970 per candidate, with no omitted unmatched/solver-failure cases. Matching
is performed on past-supported agents before future-label access, so scored-only
coverage and realized risk are not thereby equal. This does not support the
proposed joint-intervention contribution under the current proxy and policies.

## Integrity and Limits

For both families, all original forecast-error exports and ordinary control
decisions match primary evaluation exactly. This verifies exported scores and
decisions, not every intermediate tensor bitwise. Twenty-four candidate/recording
parts per family have verified source receipts. Prefix ADE has 33,686 complete
paths; FDE has 33,690 valid endpoints. Unknown full-path easy/hard labels remain
unknown. Seed variation cannot replace independent-scene uncertainty at one
historically exposed development site.

Focused comparison, supplement, error-scale, geometric-bound and replay checks:
24 passed. Previously verified training/resume checks remain valid for the same
model code. The legacy full suite was not rerun or declared green; its recorded
unrelated data-lake fixture failure remains open. No deployment upgrade, formal
safety, metric/seconds, true-3D, foundation, Stage5C or SMC claim is made.

- [CV-skip full supplementary tables](../8to12_residual_skip_v7_supplement/results.md)
- [Bounded full supplementary tables](../8to12_motion_bounded_v7_supplement/results.md)
- [CV-skip replay audit](../8to12_residual_skip_v7_supplement/replay_audit.json)
- [Bounded replay audit](../8to12_motion_bounded_v7_supplement/replay_audit.json)
