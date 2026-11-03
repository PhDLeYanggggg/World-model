# Shared Observed-Motion Cost Study

## Evidence Status

Fresh_run: complete real Torch fitting of 36 cost heads, excluded-source
cost inference and original predicted-budget joint allocations.
Cached_verified: complete saved-head prediction/scalar replay and saved
solver feasibility. Replay does not claim bitwise MILP reoptimization.
The first complete head in each feature arm has an exact independent refit.
This report only formats verified receipts; it does not fit or select.

**The original 24-group primary is not estimable and has not passed.**
The 23 supported-group results are development secondary evidence.
They are not independent geographic-site confirmation or deployment.

## Fixed Information Contrast

EqMotion was fixed before the parent forecast readout. Each cost head
uses 380 causal local/rollout columns plus 48 observed-motion slots.
Only the added 48 slots are zeroed in the local control after a common
critic-fitting normalization. Capacity, initialization, loss, source draws
and budget are paired. These are not camera pose or image features.
Producer, critic-fitting and held source groups are pairwise disjoint.

## Whole-Population Feature Effect

Shared-motion minus local-history gain, percentage points. The fixed
known reference denominator is retained; completion lower bounds account
for unknown outcomes without imputed ground-truth positions.

| Policy | Known-label paired effect [95% CI] | Completion-lower effect [95% CI] |
|---|---:|---:|
| floor | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| frame_uniform | -3.2919 [-7.1223, -0.0566] | -6.5696 [-11.1209, -2.9505] |
| independent | -1.1712 [-4.3604, +1.8679] | -4.2506 [-8.0446, -1.0018] |
| joint_free | -0.4068 [-2.6646, +2.0005] | -5.3316 [-8.9456, -2.2735] |
| joint_matched | -0.6844 [-3.2431, +1.9070] | -5.3632 [-9.1695, -2.1500] |
| uncontrolled_neural | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

These are 3,000 source-group bootstrap resamples after producer and
three actual training-seed averages, not IID-window confidence intervals.
The joint-free completion-lower contrast was fixed prospectively; other
policy views do not replace it after inspecting their scores.

## Accuracy And Original Full Risk

| Arm | Policy | Known gain (%) | Completion-lower gain (%) | Supported recording views | Known all/easy positive-harm violations | Worst known easy net degradation (%) |
|---|---|---:|---:|---:|---:|---:|
| local_history | floor | +0.0000 | +0.0000 | 0/348 | 0/0 | +0.0000 |
| local_history | frame_uniform | +5.2071 | +2.0541 | 0/348 | 143/151 | +277.9125 |
| local_history | independent | +9.7080 | +5.1618 | 7/348 | 285/280 | +309.5896 |
| local_history | joint_free | +11.3326 | +5.1034 | 7/348 | 281/283 | +340.5715 |
| local_history | joint_matched | +10.7543 | +4.7033 | 4/348 | 283/282 | +340.5715 |
| local_history | uncontrolled_neural | +16.4074 | -5.0187 | 6/348 | 304/320 | +1742.9603 |
| shared_motion | floor | +0.0000 | +0.0000 | 0/348 | 0/0 | +0.0000 |
| shared_motion | frame_uniform | +1.9152 | -1.9478 | 3/348 | 151/134 | +309.5896 |
| shared_motion | independent | +8.5368 | +3.9838 | 9/348 | 291/287 | +363.0465 |
| shared_motion | joint_free | +10.9258 | +4.7073 | 9/348 | 288/291 | +442.3645 |
| shared_motion | joint_matched | +10.0699 | +4.3405 | 7/348 | 287/288 | +363.0465 |
| shared_motion | uncontrolled_neural | +16.4074 | -5.0187 | 6/348 | 304/320 | +1742.9603 |

Recording views repeat seeds and excluded producers; they are not that
many independent places. Undefined support is not a known risk violation
and is not a safety pass. All required full-policy views retain the
unchanged 2% positive-harm and easy-degradation requirements.

## Same-Count Utility Only

Count matching is within each feature arm against its own independent
policy, not between feature arms. Eligibility is the complete query where
the causal independent intervention count is positive. Future labels do
not determine membership. Empty controls are not estimable, not zero.

| Arm | Nonempty / empty query views | Nonestimable source views | Known utility [95% CI] | Completion-lower utility [95% CI] |
|---|---:|---:|---:|---:|
| local_history | 14677/2057 | 4 | not estimable | not estimable |
| shared_motion | 15093/1641 | 3 | not estimable | not estimable |

These controls measure utility and never replace whole-population safety.
A not-estimable contrast is not silently averaged over remaining sources.

## Compute And Reproduction

Complete heads: 36; primary updates: 72,000.
Actual train invocation: 2689.130s; peak RSS: 7,393,820,672 bytes.
Native arm64 Torch CPU4, interop1, workers0. Checkpoints and solver
receipts stay private; code, configurations and light scalar receipts
are versioned. Pilot work and first-head refits are additional verification
work, not extra primary updates.

## Limitations And Decision

All-current crowd summaries may add information but are not calibrated
ego-motion. Joint allocation pools predicted budgets; it does not impose
pairwise geometric consistency or physical collision constraints.
Partial-label ADE is not complete 12-step truth. Source components are
not verified IID geographic sites. Calibration and confirmation remain
closed. No promotion follows from a positive secondary mean alone.
Image pixels/native annotation steps are not verified metres/seconds.
No true3D, foundation, successful world-model or submission-ready claim.
Deployment unchanged. Stage5C execution and SMC remain off.
