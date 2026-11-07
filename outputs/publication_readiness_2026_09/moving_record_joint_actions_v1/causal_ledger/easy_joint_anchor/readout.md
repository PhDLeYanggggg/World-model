# Can A Joint Easy-Support Head Open A Recording?

Fresh local fits, frozen forecast paths, 14 sources, 34 recordings, and
102 views across three forecaster seeds. Each of 42 cells fits on seven
sources, selects a threshold on six different sources, and evaluates
the remaining source. The direct calibration policy uses the same
seven-source coefficients as the evaluated fold. The 8/12 image-space
protocol, original 2% risk scorer, and direct decisions are unchanged.

| Opening rule | Total switches | Full passes / 102 | Easy net >2% | Zero-direct views opened / passed |
| --- | ---: | ---: | ---: | ---: |
| Direct only | 6,952 | 2 | 8 | 0 / 0 |
| Calibrated joint event | 6,955 | 2 | 8 | 0 / 0 |
| Calibrated product of marginals | 6,955 | 2 | 8 | 0 / 0 |
| Diagnostic joint, no threshold | 6,998 | 5 | 11 | 18 / 3 |
| Diagnostic original utility, same query and count | 6,998 | 3 | 12 | 18 / 1 |

Both learned rules abstain in 39/42 threshold cells. Neither adds a
pass or reaches any zero-direct view after calibration. Compared with
the marginal rule, the joint rule changes completion-lower ADE utility
by **-0.00247 pp** (95% source interval **[-0.00741, 0]**). Without the
threshold, joint ranking changes utility by **-0.00969 pp** versus
original utility at the same queries and count (95% interval
**[-0.06142, +0.02585]**). The improved risk-pass count in this
diagnostic is not a reliable utility improvement, and the increase in
signed easy violations prevents a policy promotion.

## Were There Useful Opening Actions?

A separate label-aware check scans the original 18 zero-direct views.
Each available witness is rescored using the original observed rule.
Nine have a positive, supported single switch at their first eligible
query; six have one only later (median one query later); three never
have such a switch in this bank. No oracle action is exported.

This separates three gaps. Ranking can in principle recover some
first-query failures. A policy that always opens at the first query
cannot recover the six later-only cases. The final three need different
candidate paths or a different, prospectively specified intervention
design. This is not evidence that any of those repairs already works.
Neither threshold relaxation nor replacing only the score is supported
as the whole fix. No deployment change or independent outcome readout.

## Verification

All 42 source splits are disjoint in their three roles. Native arm64
fit and calibration take 37.1 s, followed by 6.4 s scoring. All 102
direct control metrics and the full scoring replay match exactly.
86 scoped tests pass. No Stage5C or SMC.

[Analysis](analysis.json) | [Opening opportunities](opportunity_diagnostic.json)
