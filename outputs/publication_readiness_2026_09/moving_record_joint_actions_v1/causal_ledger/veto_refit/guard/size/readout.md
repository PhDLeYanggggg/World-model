# More Calibration Sources Did Not Improve Held-Source Utility

This is a fresh development experiment on exposed Moving A: 14 sources, 34
recordings and 102 recording views across three frozen forecaster seeds. It
uses 8 observed and 12 predicted annotation steps in image coordinates.
It is not independent confirmation or evidence of physical safety.

All policies use the same seven fitting sources in each fold and exactly
6,952 switches, matched query by query. The two three-source calibration sets
are disjoint within a fold; neither they nor the held source fits the cost
model. Threshold choices freeze before readout. This is not a comparison
with the ten-source fit in the preceding study.

| Paired comparison | Change in completion lower (pp) | 95% source bootstrap interval |
| --- | ---: | ---: |
| Six calibration sources vs mean of three-source A/B contrasts (primary) | -0.06765 | [-0.18612, +0.01171] |
| Pool six for utility; keep three for preservation | -0.00591 | [-0.02312, +0.00695] |
| Pool six for preservation as well | -0.06192 | [-0.17758, +0.01577] |
| Six calibration sources vs base without reopening | +0.00254 | [-0.02105, +0.02338] |

The primary mean averages paired contrasts; it does not deploy an average
policy. Completion bounds across different action disagreements need not
add. The bootstrap resamples 14 whole sources 3,000 times after averaging
forecaster seeds; overlapping queries are not independent replicates.

| Policy | Full original passes / 102 | Undefined easy risk | Easy net degradation >2% | Reopened decisions |
| --- | ---: | ---: | ---: | ---: |
| Base without reopening | 2 | 35 | 8 | 0 |
| Unguarded | 1 | 39 | 15 | 2,531 |
| Three A | 1 | 35 | 8 | 214 |
| Three B | 1 | 37 | 12 | 1,267 |
| Six for utility, A for preservation | 1 | 36 | 10 | 532 |
| Six for utility, B for preservation | 1 | 36 | 12 | 1,049 |
| Six for both | 1 | 36 | 10 | 471 |

A and B chose different thresholds in 30/42 source-seed folds. The all-six
arm selected no reopening in 23/42 folds. More utility observations did
not give a detectable benefit; applying preservation to more sources
reduced utility at the point estimate without more full held-view passes.
This does not rule out independent, better-supported calibration. It does
show that this radius guard and source expansion do not resolve the model's
utility and harm tradeoff. Keep the original 2% and support rules, including
undefined views. No deployment change, Stage5C execution or SMC.

The scoring and replay receipts have identical analysis hashes. An isolated
aggregate-only replay reproduces 29 contrasts and 17 policy summaries from
this and the two previous guard studies without raw trajectories or weights.
See [replay instructions](../../../../../../../reproducibility/calibration_evidence/README.md).
