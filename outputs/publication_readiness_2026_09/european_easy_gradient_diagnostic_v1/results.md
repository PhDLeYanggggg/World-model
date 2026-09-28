# Fitting-Only Gradient and Risk-Signal Results

Result source: fresh_run diagnostic and exact same-runtime replay on cached_verified fitting-only inputs. Registration preceded execution. There were zero optimizer updates and no new held readout. All 108 paired fits are retained; 1,728 gradient batches repeat the same source-role and seed structures, not independent data.

## Gradient Geometry

| Arm / state | Block | Auxiliary / risk norm median [10th, 90th percentile] | Auxiliary conflict / defined batches | Total conflict / defined batches | Total / risk projection median |
|---|---|---:|---:|---:|---:|
| marginal_initial | all | 0.625641 [0.182714, 1.72585] | 9/432 | 0/432 | 1.16274 |
| marginal_initial | shared | 0.594399 [0.171606, 1.7313] | 16/432 | 0/432 | 1.14479 |
| marginal_initial | output | 0.656314 [0.200726, 1.72006] | 0/432 | 0/432 | 1.2239 |
| marginal_final | all | 8630.86 [672.072, 39130.9] | 293/432 | 293/432 | -161.594 |
| marginal_final | shared | 5979.53 [453.08, 34276.6] | 302/432 | 299/432 | -181.823 |
| marginal_final | output | 10563 [877.36, 41641.1] | 273/432 | 271/432 | -129.796 |
| supervised_initial | all | 0.625641 [0.182714, 1.72585] | 9/432 | 0/432 | 1.16274 |
| supervised_initial | shared | 0.594399 [0.171606, 1.7313] | 16/432 | 0/432 | 1.14479 |
| supervised_initial | output | 0.656314 [0.200726, 1.72006] | 0/432 | 0/432 | 1.2239 |
| supervised_final | all | 176.191 [27.3168, 653.157] | 98/432 | 92/432 | 18.2101 |
| supervised_final | shared | 213.257 [29.5742, 876.134] | 66/432 | 60/432 | 27.3998 |
| supervised_final | output | 128.946 [20.817, 476.994] | 144/432 | 138/432 | 11.6059 |

Conflict means a negative cosine. A negative total/risk projection means the negative total gradient is locally uphill for direct risk under a Euclidean infinitesimal step. It does not prove that the actual finite AdamW step or held policy got worse. Gradients were not applied.

## Fitting-Source Signal

These are source-query-balanced observed-label summaries. Realized within-budget examples are not causally identifiable safe admissions or an oracle used at inference.

| Diagnostic across 216 repeated fitting-source views | Median | 10th percentile | 90th percentile |
|---|---:|---:|---:|
| easy_probability | 0.288942 | 0.155099 | 0.42213 |
| easy_signed_risk | 0.00466419 | 0.00114399 | 0.0140455 |
| easy_realized_within_budget_fraction | 0.676915 | 0.447507 | 0.838649 |
| easy_zero_harm_fraction | 0.647903 | 0.430534 | 0.808234 |
| easy_positive_harm_over_reference | 0.198489 | 0.089122 | 0.644614 |

## Source Breakdown

| Fitting locality | Views | Easy prevalence median | Easy harm/reference median | Easy realized within-budget fraction median |
|---|---:|---:|---:|---:|
| eu-locality-007 | 18 | 0.335498 | 0.36207 | 0.568948 |
| eu-locality-008 | 18 | 0.296768 | 0.171148 | 0.785114 |
| eu-locality-020 | 18 | 0.571837 | 0.556742 | 0.650242 |
| eu-locality-048 | 18 | 0.263312 | 0.107833 | 0.818589 |
| eu-locality-067 | 18 | 0.250317 | 0.0996483 | 0.697002 |
| eu-locality-074 | 18 | 0.155706 | 0.552983 | 0.551361 |
| eu-locality-082 | 18 | 0.350607 | 0.344718 | 0.586147 |
| eu-locality-110 | 18 | 0.261945 | 0.306319 | 0.644504 |
| eu-locality-112 | 18 | 0.141384 | 0.456385 | 0.558492 |
| eu-locality-119 | 18 | 0.312743 | 0.238092 | 0.753614 |
| eu-locality-124 | 18 | 0.213587 | 0.266913 | 0.692397 |
| eu-locality-126 | 18 | 0.421489 | 0.229372 | 0.672998 |

## Execution and Limits

CREATE job 37569222: 179.61 seconds, peak RSS 2,147,220 KiB. Replay job 37569306: 171.78 seconds; all group results exact. Scalar consistency checks: 5,184.

No primary or risk tolerance changed. Independent roles remain closed; no deployment, scientific efficacy, metric/seconds, human-gold, true3D, foundation or physical-safety claim. Stage5C/SMC disabled.
