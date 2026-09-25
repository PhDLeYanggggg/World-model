# Failure Taxonomy

This is post-hoc diagnosis of opened-source development data, not model or
threshold selection. Existing checkpoints and the registered 2% pointwise rule
stay fixed. Labels describe outcomes; they do not determine inference decisions.

## 1. Simple Producer-Size Mismatch: Repair Not Supported

Only 9/36 protected all-ADE replacement estimates favor the smaller producer.
Five intervals favor replacement and fifteen favor the original. Raw comparison
has no strictly positive interval among 18 distinct pairs. A systematic size
mismatch repair is therefore not supported. This does not prove the absence
of producer distribution shift: fitting identities and model quality also change.

## 2. Fitting-Roster and Candidate-Quality Sensitivity: Observed

For fitting fold1/all-event, all three seeds show opposite behavior for the two
halves. The following gains are relative to CV, not relative to one another:

| Seed | Full4 raw ADE gain (%) | Half0 raw gain (%) | Half1 raw gain (%) | Full4 controlled gain (%) | Half0 controlled gain (%) | Half1 controlled gain (%) |
|---|---:|---:|---:|---:|---:|---:|
| 17 | -2.7070 | +2.0614 | -13.6904 | +1.0777 | +1.9252 | -1.9141 |
| 29 | -2.6261 | +1.3570 | -13.1623 | +1.5444 | +2.0476 | -0.8106 |
| 43 | -2.3246 | +2.4738 | -12.9485 | +2.2989 | +3.2522 | -0.3495 |

The positive half0 controller still fails easy preservation. Selecting half0
after reading these results would not be an independent validation procedure.

## 3. Selected Harm Is Underestimated: Observed

An illustrative, not independently selected, failure slice is fold1/seed17,
all-event, locality008. The full all-view tables contain every other slice.

| Producer | Utility-head predicted / actual mean positive harm (px) | Risk-head predicted / actual selected event-harm ratio (%) |
|---|---|---|
| Full4 | 0.0442 / 18.1330 | 0.9821 / 51.7619 |
| Half0 | 0.0417 / 10.8690 | 1.0079 / 30.8679 |
| Half1 | 0.0461 / 27.4922 | 0.9940 / 78.2641 |

The estimated ratio remains near1% while realized harm differs substantially.
These ratios use event-reference mass among selected rows. They are neither
population harm rates nor net easy degradation. Utility-harm and risk-harm
heads are different fitted heads and should not be conflated.

Across all defined selected locality/view cells, actual ratios exceed2% in
43/64 full4 all-event cells, 44/64 half0 cells and 33/59 half1 cells. For
easy-event views the counts are46/72, 50/72 and30/71. Cells share localities,
so these fractions are descriptive rather than binomial confidence statements.

## 4. Missed Opportunity and Harm Can Coexist: Observed

In fold1/seed17/all-event, full4 captures4.1333% of CV-denominated benefit but
pays3.0556% harm. Half1 captures3.5958% and pays5.5100%, leaving negative net
gain. Its hypothetical oracle benefit is still18.2084%; a large oracle number
does not justify intervention when the head cannot recognize which cases help.

Fitting fold2 shows the opposite issue: good raw predictions coexist with very
low all-event switch rates. Thus the data show both excessive vetoes and harmful
accepted interventions, not merely a universally over-conservative threshold.

## 5. Candidate Sensitivity, Input Scaling or Saturation: Not Yet Identified

The head receives355 causal features, including full baseline and candidate
rollouts, their summaries, past target/neighbor motion and observed scale.
Candidate input is present; it would be incorrect to diagnose a missing rollout
field. Similar switch rates are not proof of identical row decisions or proof
that the head ignores the candidate. A controlled sensitivity check and a
matched retraining ablation are still needed to separate normalization,
output-link saturation, fitting support and feature-weight effects.

## Boundaries

The complete training/preprocessing chain excludes each pair's eight readout
localities, but all twelve sources have been used for development. No new
independent evidence, metric/seconds claim, human-gold label, physical-safety
claim, true3D/foundation claim, joint-control contribution or deployment.
No Stage5C execution and no SMC. Changing producer alone is not a safe repair.
