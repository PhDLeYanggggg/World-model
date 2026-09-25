# First Frozen Readout on Six Model-Selection Localities

## What Was Run

I evaluated the complete frozen incumbent-relative policy family on six previously
reserved model-selection localities, not on another slice of the twelve opened
training localities. Registration3d8ee3d1 and prediction freezec78fc812 were both
pushed before outcome evaluation. This is fresh conversion, inference and readout
using cached-verified checkpoints, with no new training or threshold tuning.

All28 recordings contribute38,102 targets at7,087 queries. There are21,434 complete,
15,820 partial and848 absent future-label rows. Inference retained all indexed
targets, including absent labels;37,254 rows support ADE and27,694 support the
requested final endpoint. Eight observed and twelve requested positions use raw
stride12. This is a registered engineering cohort, not a full-dataset benchmark.

## Results

The add-only controller improves source-equal all-ADE over the frozen old controller
in all36 views: +0.010488% to +0.778997%. Thirty-three locality-bootstrap intervals
are positive and none are negative. Averaging the three seeds within each locality
gives12 source/controller/event estimates;11 intervals are positive and one spans
zero. These are dependent exploratory comparisons on six localities, not36
independent tests, and the intervals are not multiplicity-adjusted.

The full family is not safe enough to promote. Seven add-only views fail the
worst-locality2% easy guard, reaching8.216907% degradation. The unrestricted neural
forecast fails easy in all36 views and harms some exact-zero-CV rows in every view.
The existing old controller itself reaches6.076432% easy degradation, so it cannot
be assumed to be a verified safety floor on these new localities.

| Frozen risk target | Add-only all gain vs old | Positive CIs | Easy-pass views | Worst easy degradation |
|---|---:|---:|---:|---:|
| All samples | +0.074892% to +0.778997% | 17/18 | 11/18 | 8.216907% |
| Positive-CV easy event | +0.010488% to +0.294365% | 16/18 | 18/18 | 0.0% |

The easy-event branch is encouraging but is not a deployment selection. It does
not uniformly beat the training-selected classical baseline: its all-ADE change
ranges from-1.773321% to+6.636237%. Its observed mean easy preservation does not mean
zero individual harm. Independent risk calibration and final confirmation have
not been performed. The predeclared no-selection rule is retained.

## Interpretation

Incumbent-relative additions show a small generalization signal beyond opened
source localities. The main remaining problem is conditional risk: a rule constrained
on the whole population can improve mean accuracy while strongly harming easy rows
in one locality. Risk-target choice matters more here than another proximity-weight
sweep. The [action accounting](failure_analysis.md) identifies the concrete losses.

No deployment, calibrated-safety or submission-readiness claim follows. All results
remain image-pixel/raw-annotation-step detector-track evidence, not metric, seconds,
human gold, true3D, foundation success or physical safety. Stage5C and SMC remain off.

## Evidence

The verification covers360 checkpoint training rosters,156 independent coordinate
array checks and4,182 metric reductions.394 tests in65 scoped files pass; the full
legacy suite was not rerun. These are engineering checks, not scientific gates.

- [Every group and three-seed summary](results.md)
- [Exact source replay](source_replay.json)
- [Data and future-truncation audit](data_audit.json)
- [Frozen decisions](decision_freeze.json)
- [Risk and failure accounting](diagnostic_accounting.json)
- [Verification manifest](verification.json)
- [Next research gap](project_gap.md)
