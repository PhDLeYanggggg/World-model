# Complete Policy Readout: Trajectory Gains, Original Risk Contract Fails

I reconstructed the entire exposed own-B population and evaluated the
frozen policies with the original per-record risk/support evaluator.
This is fresh complete-policy development evidence, not a new held test.

**Neither mean nor unanimity passes the original complete risk contract.**
No deployment upgrade. Empty support stays NA; no identity exemption.

## Complete Population

62,466 past-eligible rows, 8 source components, 24 recordings. There are 60,701
rows with some labels, 40,043 complete 12-label rows, 20,658 partial rows and
1,765 wholly unknown rows. The frozen proposer selects 49,335 rows:
48,061 known and 1,274 unknown. No future mask enters any choice.

## Full Policy Results

| Policy | Available-label ADE gain % | Completion lower gain % | Selected | Unknown selected | Supported record views | Easy reference NA |
|---|---:|---:|---:|---:|---:|---:|
| forest64 | +0.003851 | +0.003193 | 12 | 1 | 1/24 | 22 |
| mean | +1.313033 | +0.823788 | 2730 | 81 | 1/24 | 6 |
| mean_matched | +1.194201 | +0.448694 | 2730 | 95 | 1/24 | 5 |
| physical_floor | +0.000000 | +0.000000 | 0 | 0 | 0/24 | 24 |
| proposal_self | +11.936624 | +4.631900 | 49335 | 1274 | 1/24 | 0 |
| seed17 | +1.856271 | +1.039348 | 4300 | 133 | 1/24 | 6 |
| unanimity | +0.335919 | +0.305609 | 472 | 11 | 2/24 | 12 |
| unanimity_matched | +0.331892 | +0.264399 | 472 | 12 | 2/24 | 13 |

Gains are equal-source macro versus each parent training-defined floor.
ADE uses available labels, not an official complete12-step ADE estimate.
The lower gain subtracts causal path-disagreement bounds for unknown labels.
Unknown samples are not treated as safe. The baseline has undefined selected
reference mass; its 0/24 support is NA, not proof that baseline motion is unsafe.

## Actual Violations Versus Missing Support

| Policy | Known all >2% | Known easy >2% | Completion all >2% | Completion easy >2% | Easy-net >2% |
|---|---:|---:|---:|---:|---:|
| forest64 | 1 | 1 | 1 | 1 | 0 |
| mean | 9 | 15 | 18 | 17 | 4 |
| mean_matched | 11 | 15 | 18 | 17 | 2 |
| physical_floor | 0 | 0 | 0 | 0 | 0 |
| proposal_self | 22 | 23 | 23 | 23 | 19 |
| seed17 | 13 | 14 | 20 | 16 | 3 |
| unanimity | 4 | 9 | 9 | 10 | 0 |
| unanimity_matched | 5 | 6 | 9 | 9 | 0 |

Violation columns can overlap; they are not additive categories. A missing
reference is separately NA. Benefit cannot cancel positive harm in the
all/easy checks. Zero easy-net violations therefore do not imply safety.

## Paired Trajectory Contrast Versus Forest

| Policy | Known gain difference pp | Conservative completion lower pp | Exploratory source interval for lower pp |
|---|---:|---:|---:|
| mean | 1.309182 | 0.820595 | [0.126605, 1.724009] |
| unanimity | 0.332068 | 0.301100 | [0.030597, 0.644316] |

3,000 paired source-component bootstrap replicates on 8 exposed sources, not
overlapping windows. These intervals describe full-population trajectory
contrasts, not the earlier signed-decision-cost intervals or a safety guarantee.
No independent confirmation or complete three-forecaster-seed claim follows.

## Causal Matching And Verification

Same-query/action-count controls are utility-only. Their eligibility uses
frozen choices and causal utility margins; empty controls remain NA.
All old known features and decisions reproduce. One original scalar float
check failed at a batch tail; the retained v2 amendment checks training-scaled
float32 roundoff plus exact per-channel and mean/unanimity decisions. No model,
inference math, threshold or risk rule changed to pass that engineering check.

## What Remains

The risk heads were trained mainly on known outcomes and can admit censored
tracks. Consensus also discards selected easy reference mass. Next test
complete-risk donor supervision using the existing conservative unknown
envelope targets, and model reference support explicitly. Keep the same
full-policy evaluator; do not repair these failures by ignoring records,
changing 2% or selecting thresholds on test outcomes.

fresh_run: scoped reconstruction, full causal inference and risk/utility metrics.
cached_verified: weights, forecasts, original splits and known-row results.
not_run: new training, new outer C/H, independent test or deployment.
Original 24-group main NA, old failures and Stage5C/SMC prohibitions remain.
Native 8/12 image-coordinate silver labels, not verified meters/seconds,
true 3D or foundation. Source annotation-generation causality is unverified.

[Record diagnostics](record_risk.csv) | [Failure analysis](failure_analysis.md)
| [Reproduction](operation_zh.md)
