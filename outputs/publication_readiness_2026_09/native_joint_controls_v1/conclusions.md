# Native-Forecast Joint Controls: Completed Negative Mechanism Test

## Evidence Status

2026-09-21. `fresh_run`: all registered scene decisions, label readout, decision
replay and separate arithmetic/exhaustive verification. `cached_verified`: native
forecasts, nested cost heads and identity-resolved past context. `not_run`: new
training, independent risk calibration, confirmation, closed-role readout or
deployment. This source-only development experiment is not a formal test set.

Registration was pushed in commit `f4d00832` before decisions or readout. The
registered code, configuration and thresholds remain unchanged. All 175,756
forecast queries, 33 recordings, 20,932 recording/frame groups and three seeds
are retained. There are only four physical source sites, all design-exposed.
8 observed / 12 predicted sampled annotation steps; SDD annotation pixels only.

## Main Result

The primary contrast, **half-count joint minus half-count unary**, is exactly
zero percentage points in every physical scene, with a paired scene-bootstrap
interval [0, 0]. Every decision is identical. This interval reflects identical
predictions in this experiment, not statistical proof of population equivalence.
There is no demonstrated non-additive coupling contribution.

| Fixed Arm | ADE Gain vs CV | Conditional Scene CI | FDE Gain vs CV | Hard Gain | Positive-Easy Degradation | Selected Query/Seed Instances |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| CV | 0.00000% | [0, 0] | 0.00000% | 0.00000% | 0.00000% | 0 |
| Uncontrolled neural | 7.63308% | [5.96322, 9.30216] | 8.64511% | 10.65751% | 21.70968% | 527,268 |
| Full independent | 1.29190% | [0.51102, 2.30072] | 1.45715% | 1.73956% | 0.60943% | 14,576 |
| Full unary | 1.29190% | [0.51102, 2.30072] | 1.45715% | 1.73956% | 0.60943% | 14,576 |
| Full joint | 1.29190% | [0.51102, 2.30072] | 1.45715% | 1.73956% | 0.60943% | 14,576 |
| Full scene-uniform | 0.05896% | [0.00132, 0.16355] | 0.06670% | 0.07334% | 0.00184% | 180 |
| Half independent | 0.54949% | [0.06513, 1.33866] | 0.66273% | 0.70707% | 0.27566% | 4,008 |
| Half unary | 0.54921% | [0.06513, 1.33856] | 0.66242% | 0.70656% | 0.27566% | 4,008 |
| Half joint | 0.54921% | [0.06513, 1.33856] | 0.66242% | 0.70656% | 0.27566% | 4,008 |
| Half scene-uniform | 0.00000% | [0, 0] | 0.00000% | 0.00000% | 0.00000% | 0 |

Errors are averaged across seeds before computing equal-scene gains. FDE uses
the final sampled target only when available. The CI resamples four physical
sites 3,000 times, not overlapping windows as independent observations. Hard
and positive-easy are complement-training-CV q75/q25 diagnostic slices, not
replacements for strict exact-zero-CV protection.

Full-count equality is a structural null: the entire eligible set is the only
feasible exact-count subset. Half scene-uniform is also structurally unable to
replace all targets at half the eligible count; its zero gain is not learned
safety. Full versus half is a capacity change, not a coupling ablation.

## What Actually Changed

All 62,796 scene/seed queries satisfy the registered count and predicted-harm
constraints; no solver failure or dropped query produces the null result.
Half-count has 3,105 nonzero-count queries. Only 88 queries (0.1401% of all
scene/seed queries, 44 unique recording/frame groups across six recordings)
have a potentially non-additive supported pair at count >=2. Most are in
deathCircle: 84 of 88 repeated instances.

Separate exhaustive enumeration covers all 88, with 12,783 candidate subsets
and 9,967 feasible subsets. The cached unary and joint solutions are optimal
under direct objective arithmetic in every case. Only one query has a unique
feasible assignment. Thus the harm constraint alone does not explain all nulls.
78 queries have variable product costs above numerical noise. In 82 queries,
the complete feasible product-cost range is smaller than the unary runner-up
gap, so it cannot overturn that ordering. The remaining five multi-feasible
queries also have the same unary/joint optimum. This explains the null for this
fixed pool, geometry and weight; it does not prove that all joint models fail.

Unary geometry changes the reference identities in only nine queries. The known
edge-proximity excess sum falls 5.9343%, but equal-scene ADE changes by
**-0.00028424 pp**, conditional CI [-0.00112695, +0.00027424]. A smaller designed
proximity proxy is not better forecasting, fewer real collisions or physical safety.
No weight or threshold is retuned after observing this result.

## Source and Seed Breakdown

| Source Site | Uncontrolled ADE Gain | Full Independent | Half Independent | Half Joint |
| --- | ---: | ---: | ---: | ---: |
| coupa | 8.64721% | 0.23511% | -0.01096% | -0.01096% |
| deathCircle | 5.06856% | 2.80531% | 1.71954% | 1.71991% |
| gates | 6.85945% | 1.34024% | 0.29338% | 0.29338% |
| hyang | 9.95711% | 0.78694% | 0.19600% | 0.19450% |

Full-independent ADE gains for seeds 17/29/43 are 1.52165%, 1.29118%, 1.06287%.
Half-joint gains are 0.55957%, 0.52511%, 0.56295%. No favorable seed is selected.
Source p95/p99 errors and all per-seed FDE records remain in `analysis.json`.
The half-count rule slightly worsens coupa rather than preserving every source.

## Safety and Missingness

The uncontrolled neural model harms 9,000 repeated complete zero-CV query/seed
instances. All gated arms harm zero observed complete zero-CV instances, because
they remain subsets of the fixed conservative proposal pool. Joint optimization
does not create an additional protection result.

Full control still selects 283 unknown-ADE and 3,064 incomplete-future instances;
half control selects 98 unknown-ADE and 891 incomplete-future instances. Partial
outcomes contribute only their observed ADE points. No future mask controls
inference eligibility, and absent outcomes never become zero-cost labels.
21,897 repeated graph edges have unsupported context forecasts, across 10,953
scene/seed queries with unsupported context. They are unpriced, not safe.
The relation of missing outcomes to intervention harm is unverified.

## Verification and Research Decision

All 5,272,680 Boolean arm choices replay exactly. Separate code verifies 920
scene reductions, fixed-count anchors, label support, seed-averaged FDE and all
88 non-additive real-query optima. This is independent arithmetic by the same
agent, not external scientific replication or independent confirmation data.
Analysis SHA256:
`a3d2087c0415d13baceb9a6ae5eea87df96619277847cfe2ca86b4560d1d6113`.

Keep the simple full control as a developmental reference, not a certified new
deployment. Do not spend another broad architecture sweep or post-hoc pair-weight
grid to defend the joint module. Prioritize reliable baseline-relative gain/harm
learning and independent support: the useful neural predictor exists, but most
of its gain is still lost under strict protection. Joint coupling remains a
negative control unless a new training-supported mechanism earns a new registered
test. Calibration/confirmation and competitive matched forecasters remain open.

Submission readiness remains unmet. No metric/seconds/true-3D/foundation claims.
No Stage5C, SMC or new deployment. See `execution_notes.md` for exact replay.
