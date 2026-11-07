# Decision Supplement v14

Fresh attribution on frozen, exposed development predictions. The
original scoring aggregates for the replayed policies and utility
contrasts match exactly. No model, threshold, risk rule, or deployment
choice changed. No DroneCrowd data or scores were accessed by this
audit.

| Policy | Switches on wholly unlabeled rows | Pass | Observed fail | Debit only | Undefined only | Debit and undefined |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gain ranking | 133 | 4 | 93 | 0 | 5 | 0 |
| Support first | 650 | 0 | 95 | 3 | 4 | 0 |
| Risk shortfall | 164 | 2 | 95 | 2 | 3 | 0 |
| Student, fixed path | 500 | 0 | 92 | 1 | 6 | 3 |
| Student, full bank | 1,394 | 0 | 98 | 1 | 3 | 0 |
| Screened student | 0 | 0 | 0 | 0 | 102 | 0 |

Observed failures take precedence: at least one original check
already exceeds 2% on observed labels. Debit-only failures exceed
the rule only after the bound for wholly unlabeled switches is added.
Undefined checks lack positive reference support. Outcome columns are disjoint;
the first five policies have identical per-query counts totaling
9,675 switches, and the last has zero. The [scalar audit](risk_attribution.json)
also records the individual checks and their overlapping patterns.

Against gain ranking, full-bank utility has an observed contrast of
-1.501 pp (95% source-bootstrap CI [-3.865, +0.353]) and a point
completion envelope of [-16.607, +13.606] pp. That envelope is not
a confidence interval; it says neither sign is established for all
missing-label completions by these bounds. The lower bound is negative
in all 14 source averages and under every single-source omission
([-17.825, -13.128] pp). For the fixed-path student, available
labels favor imitation by +0.677 pp [0.030, 1.506], but its point
completion envelope [-2.020, +3.375] pp also spans both signs.
Missing-label uncertainty matters for utility; it does not explain
away the observed recording failures.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_structured_risk_attribution.py
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v14.py
cd data/stage_cvpr2027_experiments/evidence_manuscript_v14/supplement
tectonic -X compile supplement.tex
```

The audit uses existing private frozen development predictions and
labels. The supplement builder needs the verified v12 aggregate
archive and CVPR style, not the prediction arrays. The task remains
observation-8/prediction-12 in image pixels, not metric or seconds.
An empty-directory rebuild reproduces all four PDF page rasters;
nine focused tests passed.
The supplement is a companion to the working draft, not independent
confirmation or a submission package.
