# Completion Screening: Tiny Coverage Gain, No Learned or Safe Upgrade

Result source: **fresh_run** source choices, causal action freeze and internal
transfer readout. Frozen forecasts, cost-head checkpoints and features are
**cached_verified**. New neural training and independent confirmation: **not_run**.

## Main Comparison

| Completion screen versus parent unknown-label veto | ADE improvement | Nominal 95% locality interval |
|---|---:|---:|
| Primary, own intervention coverage | +0.0304981% | [+0.0106989%, +0.0515714%] |
| Same per-query intervention count | 0.0% | [0.0%, 0.0%] |

The 3,000-draw bootstrap aggregates dependent seed/view results within12
development localities first. These are repeatedly exposed development data,
not independent confirmation or a multiplicity-corrected significance claim.
The zero matched contrast follows identical actions in all216 views. It is not
a statistical equivalence result or evidence that an abstaining policy learned.

| Policy | Parent | Completion screen |
|---|---:|---:|
| ADE gain against floor | +0.0038213% | +0.0343193% |
| Hard ADE gain against floor | +0.0000876% | +0.0009585% |
| Mean intervention rate | 0.2635933% | 2.8466638% |
| Worst whole-population easy ADE degradation | 0% | 0% |
| Defined selected-risk views | 9 | 27 |
| Undefined selected-risk views, not passes | 207 | 189 |
| All selected-risk violations | 0/9 | 3/27 |
| Easy selected-risk violations | 2/9 | 5/27 |
| Worst easy selected positive-harm ratio | 3.04230% | 4.53891% |
| Selected unknown outcome occurrences | 53 | 498 |

The quantities in the easy rows are different. Overall easy ADE can improve
while selected positive harm exceeds budget; benefits do not cancel that risk.
Unknown occurrence counts include repeated contexts, not independent people.

## What Was Selected

The source rule selects63 fallback, six explicit initial-prior heads and three
MSE-selected heads at stepzero. **No trained checkpoint wins.** Relative to the
parent, six additional source-fit policies intervene; three unchanged sources
retain identical prior decisions. The improvement is a coverage effect of
previously frozen forecasts under source priors, not new neural dynamics,
representation learning or better within-query ranking.

On transfer, the conservative missing-outcome screen supports11/27 nonempty
views. The other16 have easy-risk upper bounds above budget; six also exceed
the all-risk upper bound. The remaining189 empty views have undefined selected
reference and are not safe passes. These transfer bounds are offline diagnostic
readouts, not used to remove rows, tune thresholds or select deployment.

The largest repeated new known-risk failure is locality008 -> locality020 in
all three seeds. Easy selected positive-harm ratios are3.24612%,4.53891%,4.41526%,
despite positive total and easy ADE gains. The other two violations are inherited
locality124 -> locality007 priors. Source-validation finite-completion support
does not imply cross-locality risk transport.

## Verification and Verdict

All72 source choices replay exactly. All216 parent action hashes match the
original control; new actions were committed before outcome readout. Independent
checks cover27,000 metric values,7,560 completion-bound aggregates and747,900
query-count matches. Full-readout replay and scoped-test receipts accompany
this report. Initial evaluation took119.24 seconds; no new checkpoint or row
cache was produced.

**No deployment promotion; no new learned contribution; not submission-ready.**
The bound resolves part of the missing-outcome support problem but does not
solve the known-outcome risk-learning and source-transfer failures. It cannot
justify reopening independent selection/calibration/confirmation or claiming
that the broader world-model goal is achieved.

Units remain image-local detector-silver, obs8/pred12 at stride12 raw frames.
No metric/seconds, true3D, foundation, human-gold or physical-safety claim.
Stage5C execution and SMC remain disabled.
