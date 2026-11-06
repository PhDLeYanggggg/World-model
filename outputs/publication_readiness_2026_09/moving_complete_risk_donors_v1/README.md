# Complete-Risk Donor Supervision

I added the conservative unknown-label contributions omitted by known-only
donor fitting. All 24 new Torch heads and 24 full exact refits are complete.
The same features, source exclusions, network, budget and zero threshold remain.
This is exposed-development model repair, not a new independent test.

## Complete Policy Comparison

| Training / policy | Available-label gain % | Completion lower gain % | Selected | Unknown selected | Original supported views |
|---|---:|---:|---:|---:|---:|
| known_only / forest64 | +0.003851 | +0.003193 | 12 | 1 | 1/24 |
| known_only / seed17 | +1.856271 | +1.039348 | 4300 | 133 | 1/24 |
| known_only / mean | +1.313033 | +0.823788 | 2730 | 81 | 1/24 |
| known_only / unanimity | +0.335919 | +0.305609 | 472 | 11 | 2/24 |
| complete_risk / mean | +1.134321 | +0.921428 | 3359 | 68 | 1/24 |
| complete_risk / seed17 | +1.804856 | +1.513273 | 4420 | 89 | 0/24 |
| complete_risk / seed29 | +1.725925 | +1.393459 | 5111 | 102 | 0/24 |
| complete_risk / seed43 | +1.687552 | +1.384762 | 5013 | 90 | 0/24 |
| complete_risk / unanimity | +0.244203 | +0.211943 | 461 | 10 | 1/24 |

Equal-source macro gain versus the original training-defined floor.
Available-label ADE is not complete-12 official ADE. Completion lower gain
retains the original unknown-label path envelopes. Empty support stays NA.

## Original Risk And Support

| Policy | Known all >2% | Known easy >2% | Completion all >2% | Completion easy >2% | Easy reference NA |
|---|---:|---:|---:|---:|---:|
| mean | 14 | 19 | 19 | 19 | 4 |
| seed17 | 19 | 20 | 24 | 21 | 3 |
| seed29 | 14 | 18 | 18 | 19 | 4 |
| seed43 | 18 | 17 | 22 | 20 | 4 |
| unanimity | 9 | 8 | 12 | 8 | 14 |

## Paired Change From Known-Only Consensus

This uses the envelope between the two policies, not subtraction of their
separate pessimistic lower gains. A higher individual lower bound is not
proof that the paired improvement is positive.

| Policy | Known pp | Completion lower pp | Exploratory source-bootstrap interval for lower pp |
|---|---:|---:|---:|
| mean | -0.178712 | -0.783298 | [-2.085325, +0.095371] |
| unanimity | -0.091716 | -0.154287 | [-0.441157, +0.120787] |

3,000 paired resamples of 8 source components; not windows, independent
confirmation or safety intervals. All policies and seed slices are retained.

## Observed Versus Bound Target Error

| Label role | Known-only MSE | Complete-risk MSE | Improved source/seed views |
|---|---:|---:|---:|
| observed | 1.715678 | 23.081753 | 0/24 |
| bound | 586.586965 | 527.056177 | 21/24 |

MSE uses each frozen known-only fitting RMS for a common comparison scale.
Unknown-role MSE measures prediction of a conservative bound contribution,
not error against an observed future. The new fitting RMS is inflated by
median factors 2.333 (all) and 5.917 (easy).
This is consistent with large bound targets disrupting observed-risk
regression, not proof that normalization alone caused the failure.

## Decision

Original complete-risk screen: **No new policy passes.**
No deployment upgrade follows this exposed-development result. The original
2% rules, support/NA semantics and old failures remain. No identity exception.

## What The Treatment Changes

Known features and targets match exactly before fitting. New unknown targets
are conservative bounds, not ground-truth future error. Rebalancing complete
source/query pools and fitting-only normalization are part of the treatment.
The comparison does not isolate those mechanisms. The unknown-role mask is
used for training/reporting only, never inference or a query rejection gate.
The existing two-channel head still does not directly ensure selected easy
reference support; stronger fitted scores alone may therefore be insufficient.

fresh_run: complete-risk supervision, 24 fits, 24 exact refits and full readout.
cached_verified: known donors, forecasts, proposer and known-only controls.
not_run: new outer C/H, independent validation, deployment or forecaster seeds.
Original 24-group main remains NA. Native obs8/pred12 image coordinates and
silver annotations; not verified metric/seconds, true 3D or foundation.
Source annotation-generation causality remains unverified. Stage5C/SMC off.

[Failure analysis](failure_analysis.md) | [Reproduction](operation_zh.md)
