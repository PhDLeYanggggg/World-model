# Fitting-Only Utility and Risk Diagnosis

fresh_run: all108 groups, 5,741,442 repeated fitting rows, 318,969 unique row IDs. Unique rows still overlap in recordings and are not independent units. Inputs and checkpoints cached_verified. Full pass 242.37s; exact full replay 248.36s.

No new training, joint optimization, threshold selection or held/independent readout. The frozen independent sign screen is a diagnostic, not a replacement deployment policy.

## Source/Query-Balanced Fitting Costs

| Frozen arm | Retained benefit | Positive harm | Net gain | All risk violations/defined | Easy violations/defined |
|---|---:|---:|---:|---:|---:|
| raw | 0.00235616 | 0.00026974 | 0.00208642 | 34701/218994 | 33951/192193 |
| trainable | 0.00184547 | 0.00021591 | 0.00162956 | 27236/149334 | 26272/132489 |
| fixed | 0.00213300 | 0.00023854 | 0.00189446 | 27894/153999 | 26852/136290 |

Costs are normalized by each head's original fitting cost scale; entries are equal-context means. They are not ADE percentage improvements. Query counts are dependent occurrences, not independent trials.

## Missed Benefit Accounting

| Exclusive rejection reason | Raw missed benefit share | Trainable | Fixed |
|---|---:|---:|---:|
| not_moving | 0.001% | 0.001% | 0.001% |
| unsupported | 3.188% | 3.188% | 3.188% |
| nonpositive_utility | 15.441% | 15.441% | 15.441% |
| all_risk_rejected | 75.330% | 75.330% | 75.330% |
| easy_risk_rejected | 4.895% | 5.144% | 5.004% |
| admitted | 1.145% | 0.897% | 1.036% |

Shares use all available fitting benefit as denominator and sum to100%, including admitted benefit. The exclusive priority order is registered; overlapping causes must not be interpreted causally.

| Raw-screen partition | Benefit | Harm | Net |
|---|---:|---:|---:|
| not_moving | 0.00000178 | 0.00000120 | 0.00000058 |
| unsupported | 0.00656005 | 0.00380641 | 0.00275364 |
| nonpositive_utility | 0.03177773 | 0.05946953 | -0.02769180 |
| all_risk_rejected | 0.15503249 | 0.05765557 | 0.09737692 |
| easy_risk_rejected | 0.01007495 | 0.00149102 | 0.00858393 |
| admitted | 0.00235616 | 0.00026974 | 0.00208642 |

Missed benefit is not recoverable gain: an excluded population can contain some beneficial switches and still have negative net gain or unacceptable positive harm. Removing a gate requires its own evidence and is not licensed by the oracle.

## Within-Query Utility Ordering

| Arm | Informative query occurrences | Utility ranking misses oracle | Utility-only worsens screen | Incomplete | Zero count | No choice |
|---|---:|---:|---:|---:|---:|---:|
| raw | 168323 | 127295 | 54500 | 91044 | 478920 | 9613 |
| trainable | 113800 | 79096 | 35795 | 91044 | 533497 | 9559 |
| fixed | 117225 | 81742 | 36919 | 91044 | 529553 | 10078 |

At the sign screen's own count, ranking the same eligible pool by realized signed gain gives an unconstrained oracle upper bound. It ignores risk constraints and is never an inference input. Only complete-label queries contribute. Utility-only ordering is not assumed safe.

## Boundaries

These scores were fitted on the diagnosed sources: this is capacity/objective debugging, not validation, calibration or evidence of generalization. Per-locality entries are in summary.json. Unknown-selected and zero-denominator query counts remain explicit. No formal risk certificate, new world-dynamics lift or deployment upgrade. Image-local detector silver, obs8/pred12, stride12 raw frames; no metric/seconds/true-3D/foundation claim. Stage5C and SMC remain off.
