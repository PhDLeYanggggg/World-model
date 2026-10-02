# Relative Leaf Targets Do Not Repair Conditional Risk

## Material Passport

Execution and validation completed: 72 unique source-only terminal-value refits,
with exact second fitting/inference/readout passes. Original trees, features,
reference predictions and source partitions are cached-verified. New fitted
values and evaluation are fresh computations. No new tree splits, neural updates,
directional transfer or independent confirmation. This is not deployment evidence.

## Main Result

| Source-held diagnostic | Original forest | Relative leaf refit |
|---|---:|---:|
| Selected occurrences | 95,455 | 93,616 |
| Selected unknown outcomes | 918 | 995 |
| Complete finite-support passes /72 | 33 | 26 |
| Defined selected easy-risk views | 43 | 60 |
| Easy completion-upper violations | 7 | 34 |
| Worst selected easy completion upper | 5.4058% | 10.6025% |
| Known-label easy-risk violations | 4 | 21 |
| Additional violations from unknown completion | 3 | 13 |

Ten previous passes are lost; three new passes appear. Twenty-three passes and
36 failures remain unchanged. Known-label violations also increase: missing
outcomes are not the sole problem. Empty/undefined sets remain failures, not
zero-risk passes. The 2% selected positive-harm/reference budget is unchanged
and differs from whole-easy net degradation.

## Paired Estimates

Intervals use 3000 bootstrap draws over 12 exposed development localities,
averaging dependent heads/controllers within locality first. They are nominal
exploratory intervals, not confirmatory tests or independent end-to-end seeds.

| Refit minus original | Mean | Nominal 95% locality interval |
|---|---:|---:|
| Normalized signed-score MSE, lower is better | +0.0733913 | [-0.0079319, +0.1788780] |
| Conservative utility / full known reference mass (%) | -0.0228841 | [-0.0545382, +0.0041222] |
| Same-query count-matched utility / full known reference mass (%) | -0.0153823 | [-0.0346704, +0.0002971] |

No registered endpoint has interval-supported improvement. The negative utility
point estimates are not statistically established degradations. These contrasts
are not ADE/FDE trajectory-improvement percentages.

The matched comparison selects 81,260 occurrences per policy, equal within each
recording+frame query, ranking by each policy's own predicted utility rather
than oracle labels. Complete support is 29 vs 30, with 11 vs 10 risk violations.
This isolated difference does not rescue the utility result or justify tuning
another threshold on these outcomes.

## Mechanism

Training membership/targets match original hashes. Weighted means in 912,895
populated leaves reconstruct original values. All 72 partitions are whole-
recording hash splits. There were zero numerical target clamps. Reference
predictions are bitwise unchanged on every evaluated row.

Scale mixing exists: the training leaf's ensemble-average envelope exceeds the
query envelope in 80,560/95,455 original selected occurrences and 6,675/7,686
observed unsafe original selections. But expressing costs as envelope fractions
inside the same learned routing is not a successful correction.

The refit adds 17,733 switches and removes 19,572. Among known-label rows,
5,591/17,486 additions are harmful (31.97%), versus 1,907/19,402 removals (9.83%).
These are descriptive occurrence rates, not independent samples or causal effects.

## Decision

**Advancement gate: false.** Do not run directional transfer or deploy this model.
The result does not support another target-rescaling-only repair. It does not
prove irreducible feature insufficiency, label noise as the sole cause, or that
all differently trained partitions would fail. Preserve the negative result
without presenting it as a world-dynamics contribution.

Scope: obs8/pred12 raw stride12, image-local detector-silver; no metric/seconds,
human-gold, physical-safety, true3D, foundation or CVPR-readiness claim.
Independent selection/calibration/confirmation stay closed. Stage5C/SMC off.
