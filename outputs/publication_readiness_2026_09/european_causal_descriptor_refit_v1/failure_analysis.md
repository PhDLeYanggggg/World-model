# Causal-Descriptor Failure Diagnosis

Post-readout, descriptive analysis of hash-verified aggregates. This does not replace the registered primary, choose a model, or change an action.

## Paired Score Error

New minus signed-excess control; negative is better. Same 3,000 locality-bootstrap draws, averaging dependent views within each of 12 opened localities.

| Quantity | Paired difference [95% exploratory CI] |
|---|---:|
| all_signed_MSE | -0.0026 [-0.0062, -0.0001] |
| easy_signed_MSE | 0.0001 [-0.0000, 0.0003] |
| all_fit_MSE | -0.0020 [-0.0025, -0.0015] |
| easy_fit_MSE | 0.0000 [-0.0000, 0.0000] |
| all_generalization_gap_difference | -0.0006 [-0.0040, 0.0019] |
| easy_generalization_gap_difference | 0.0001 [-0.0000, 0.0003] |

Training-fit improvement is not held-source calibration. The four output components are not separately identified cost moments.

## Same-Query Selection Quality

| Held development locality | ADE advantage over count-matched control (%) |
|---|---:|
| eu-locality-007 | 0.009014 |
| eu-locality-008 | -0.018612 |
| eu-locality-020 | 0.000000 |
| eu-locality-048 | -0.008369 |
| eu-locality-067 | 0.002378 |
| eu-locality-074 | 0.006389 |
| eu-locality-082 | 0.002199 |
| eu-locality-110 | -0.088920 |
| eu-locality-112 | -0.004974 |
| eu-locality-119 | -0.017527 |
| eu-locality-124 | -0.016709 |
| eu-locality-126 | -0.031383 |

Each query fixes the same intervention count, eligible pool and row-ID tie break. This is an ordering diagnostic, not a new deployable rule.

## Risk Failures

| Policy | Worst defined harm (%) | Known selected rows in that view | Empty views |
|---|---:|---:|---:|
| control | 51.617505 | 5 | 14 |
| descriptor | 43.733960 | 1 | 10 |
| control_matched_count | 43.733960 | 1 | 10 |

View counts are dependent source/producer/fit/seed views, not independent scenes. Empty denominators stay undefined. Unknown-label actions cannot be claimed safe.

## Interpretation

The added descriptors change intervention coverage and slightly improve average error, but do not improve equal-budget ranking. Easy net preservation coexists with positive-harm budget violations. This falsifies the simple descriptor-only repair in this setup, not the usefulness of all causal motion features.
The evidence localizes a decision-quality and conditional-risk problem; it does not establish whether loss identifiability, imperfect utility scores, label noise or source shift is the sole cause. The 384 additional parameters also prevent an equal-capacity semantic claim.
Next: freeze these actions and decompose benefit versus positive harm in the selections that differ from the same-query control, including unknown-label and sparse-view support. Then register one targeted gain/harm ordering repair. Do not loosen the budget or pick a favorable held slice.

Image-local detector silver, obs8/pred12 rawstride12. No independent confirmation, deployment, metric, seconds, physical-safety, true3D or foundation claim. Stage5C and SMC remain disabled.
