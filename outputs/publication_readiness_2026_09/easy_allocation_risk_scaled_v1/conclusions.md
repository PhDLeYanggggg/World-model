# Query-Level Risk Allocation: Utility Recovery Without a New Best Policy

2026-09-24. Complete, numerically repaired and replay-verified development study.
`fresh_run`: fixed query allocation, numerical diagnosis/repair and metric readout.
`cached_verified`: all forecasters, gain/harm heads, 36 easy-moment forests,
source exclusions, coordinates and labels. New training, external prediction,
independent calibration and confirmation are `not_run` in this experiment.

## Question and Protocol

The previous pointwise easy-risk rule rejected nearly all useful neural
intervention. We tested whether pooling predicted risk within the same
recording/frame can recover utility, and whether nonadditive geometry improves
prediction beyond unary geometry at the same count and risk budget.

All 175,756 past-eligible target windows from 33 recordings and four exposed SDD
sites were retained, with seeds 17/29/43 and three frozen predictors. This gives
188,388 query/action/seed instances, not that many independent scenes. There are
143,918 complete futures, 29,039 partial futures and 2,799 with no future labels;
future availability never determines inference support.

Primary descriptive metric: equal-site relative available-point ADE gain over
constant velocity (CV), obs8/pred12 native annotation steps at stride12, annotation
pixels. Bootstrap uses 3,000 paired physical-site draws over only four exposed
sites. Seed-averaged errors are not a deployed prediction ensemble. These are
not historical raw-frame t+50 scores, seconds, metric, untouched confirmation,
true 3D, foundation-model or physical-safety results.

## Results

| Predictor / rule | ADE gain % | Hard gain % | Worst site/seed easy degradation % |
|---|---:|---:|---:|
| Damping / old strict | 3.6347 | 4.7232 | 2.4944 |
| Damping / pointwise | 0.1241 | 0.0004 | 0.0000 |
| Damping / selected-set budget | 0.1913 | 0.0022 | 0.0000 |
| Damping / population budget | 0.9213 | 0.6229 | 0.0000 |
| Transformer / old strict | 2.4368 | 2.2918 | 1.0669 |
| Transformer / pointwise | 0.0134 | 0.00005 | 0.0000 |
| Transformer / selected-set budget | 0.0170 | 0.0001 | 0.0000 |
| Transformer / population budget | 1.2802 | 0.6425 | 0.1120 |
| Transformer / matched unary | 1.2781 | 0.6412 | 0.0954 |
| Transformer / matched joint | 1.2781 | 0.6412 | 0.0954 |
| EqMotion / old strict | 1.6093 | 0.5271 | 0.4522 |
| EqMotion / pointwise | 0.0013 | 0.0000 | 0.0000 |
| EqMotion / selected-set budget | 0.0013 | 0.0000 | 0.0000 |
| EqMotion / population budget | 0.9556 | 0.2504 | 0.0381 |

All 27 predictor/rule combinations, FDE, intervention rates, incomplete outcomes
and site/seed details remain in [results.md](results.md), [results.csv](results.csv)
and [site_seed_results.csv](site_seed_results.csv). Uncontrolled and whole-scene
controls are retained; this table is not a winner-selection step.

Population-budget ADE CI95: damping [0.4882, 1.5333]%, Transformer
[0.5647, 2.1160]%, EqMotion [0.3536, 1.6291]%. Transformer population minus old
strict is -1.1567 percentage points, paired CI95 [-2.0064, -0.5223]. EqMotion's
corresponding difference is -0.6537 pp [-1.3861, -0.1398]. Lower observed easy
degradation therefore comes with materially lower utility, not a new best policy.
The old strict damping rule itself exceeds the 2% easy ceiling and is not a
protected deployment alternative.

## Mechanism and Negative Evidence

1. Pooling only selected targets has little effect. Selected-set minus pointwise
   gains are +0.06721 pp for damping, +0.003562 pp for Transformer and exactly
   zero for EqMotion. Hard gain remains negligible. This does not repair the
   practical conservatism of the selected-set easy-risk target.
2. Most recovery comes from including all forecastable targets in the denominator,
   including targets that retain CV. This is an explicitly registered change in
   risk accounting, not proof of learned multi-agent interaction. No allowance
   is borrowed from another query, recording, site or non-target context agent.
3. The population rule selects 31,783, 56,730 and 23,323 target/seed instances
   for damping, Transformer and EqMotion. Unary and joint use exactly these counts
   in every query after numerical repair. Whole-scene uniform selects only 75,
   39 and zero, under its explicitly stricter all-target common-support rule.
4. Nonadditive opportunity exists in 150, 1,253 and 387 query/seed instances, but
   joint changes the unary identities in only 2, 20 and 3 queries respectively.
   All counts are out of 62,796 query/seed instances per predictor. Transformer
   joint minus unary is only +0.00004811 pp, CI95 [0, 0.00009622]; its hard contrast
   is slightly negative. Damping and EqMotion point contrasts are negative and
   their intervals include zero. There is no substantial predictive interaction
   contribution to promote.
5. Joint reduces the chosen proximity proxy slightly; this is not a measured
   collision reduction or physical-safety result. Transformer joint also loses
   0.00204 pp against population allocation, despite an identical intervention
   count. A lower optimization objective does not imply better trajectory ADE.
6. The Transformer population/unary/joint rules harm two complete zero-CV
   query/seed instances. Positive-easy percentage degradation does not expose
   these cases because its baseline denominator is zero. They remain separately
   reported; do not turn the small positive-easy average into a no-harm guarantee.

## Numerical Failure and Repair

V1 correctly rejected 127 numerically invalid query solutions and fell back to CV.
Its small, unscaled risk coefficients were within HiGHS's absolute feasibility
tolerance, although the rounded solutions exceeded the original-unit postcheck.
Fourteen causal-only reproduced failures all show risk overrun, not a wrong
integer or pair product. This is an implementation issue, not a training failure.

The versioned repair scales gain, harm, geometry weight and risk budget by the
same positive factor; the original feasible set and objective ordering are
unchanged. Only the 127 failed queries are recomputed. All 188,261 successful
query decisions are preserved exactly, and 877 agent/arm bits change in repaired
queries. Original V1 source, decisions and results remain archived. No new
threshold, label-driven repair subset, model fit or objective change is involved.

All repaired query controls now pass numerical checks and exact count matching;
there are zero unmatched queries and zero predicted-budget violations for the
new rules. That verifies numerical constraints, not calibration of predicted risk.

## Reproducibility and Decision

Exact decisions/aggregate replay and separate raw-label arithmetic pass. The
verification recounts 14,236,236 decision bits, 1,190 small-query optima, 2,592
scene reductions and 45 paired contrasts. Seventeen preselected checks exceed
the ten-variable brute-force limit and are explicitly skipped; this is not
exhaustive proof of every large MILP. Allocation/query function ASTs match V1,
with only the solver dependency replaced. 115 scoped tests pass; the unrelated
legacy suite was not rerun. Verification is a second implementation by the same
research workflow, not external replication. See [execution_notes.md](execution_notes.md).
An independent code review also prompted a guarded operational replay entrypoint:
complete frozen-parent and repair-identity equality are now enforced without
rewriting the historical experiment source or its metrics.

**Do not promote the new rules.** They recover some utility from the failed
pointwise rule, but neither surpass the old protected neural comparator nor
establish meaningful nonadditive-agent lift. The historical deployable artifacts
are not changed; their old metrics are not made comparable by this experiment.

The next useful diagnosis is source-excluded reliability of the predicted easy
numerator/denominator and its support, including complete-label selection and
zero-CV cases. This must precede a newly registered target/representation repair,
not a sweep of thresholds on these exposed results. A new independent calibration
source remains necessary. Dense-source admission requires camera/site provenance;
HT21 stays quarantined, DUT diagnostic, DroneCrowd confirmation closed.

No new deployment, world-model success, submission readiness, Stage5C or SMC.
Analysis SHA256: `73dc1e392efe1bc6b61569d474e09158284118da13bebc9625ce317e078a6956`.
