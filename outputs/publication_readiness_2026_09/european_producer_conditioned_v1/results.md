# Producer-Conditioned Controllers: Complete Development Readout

I compared three newly trained, equal-capacity gain/harm controllers on identical two-source forecasts: no tag, actual producer tag, and an outcome-independent placebo tag.
All 108 neural heads reached their registered 2,000 updates (216,000 total); no new trajectory forecaster was trained.
All 180 views are retained. Each policy spans 36 dependent fold/seed/event/producer views, not 36 independent tests.
The predictor source tag is confounded with the fitting cohort. Even a positive tag effect would not uniquely identify producer transport.

## Identical-Forecast Primary Comparisons

| Producer vs control | All ADE gain (%) | Positive / negative CI | Hard gain (%) | Complete-label gain (%) | FDE gain (%) |
|---|---:|---:|---:|---:|---:|
| global | -0.5026 to 0.9393 | 13 / 9 | -0.5270 to 1.2540 | -0.6001 to 1.1766 | -0.6547 to 1.3495 |
| placebo | -0.3266 to 1.0203 | 10 / 9 | -0.3653 to 1.3210 | -0.4711 to 1.3837 | -0.4296 to 1.4851 |
| wrong_tag | -0.8585 to 1.5225 | 12 / 9 | -1.1338 to 2.1330 | -1.0080 to 1.9283 | -0.9519 to 2.0249 |
| legacy | -0.5108 to 1.0147 | 13 / 6 | -0.4824 to 1.2039 | -0.5862 to 1.2683 | -0.7275 to 1.4747 |

## Common Anchors And Safety

The four-source floor and unchanged stopping controller are common anchors. Gains over each branch's two-source floor are reported separately, never substituted as the main denominator.

| Policy | All gain vs common floor4 (%) | All gain vs old stop4 (%) | Positive / negative CI vs stop4 | Worst positive-easy degradation vs CV (%) | Zero-CV harmed views | Switch rate |
|---|---:|---:|---:|---:|---:|---:|
| global | -0.7170 to 2.5587 | -1.1465 to 2.2622 | 15 / 12 | 7.98110 | 0 | 0.0045 to 0.3880 |
| producer | -0.7529 to 2.5726 | -1.0378 to 2.2763 | 15 / 10 | 8.02939 | 0 | 0.0044 to 0.5370 |
| placebo | -0.6939 to 2.5630 | -1.1277 to 2.2666 | 15 / 12 | 7.85380 | 0 | 0.0047 to 0.3815 |
| wrong_tag | -0.7973 to 2.5527 | -1.4212 to 2.2562 | 15 / 9 | 8.13703 | 0 | 0.0049 to 0.4776 |
| legacy | -0.7691 to 2.5705 | -1.0540 to 2.2742 | 15 / 12 | 8.19116 | 0 | 0.0044 to 0.3831 |

## Raw Producer Change

| Comparison | All ADE gain (%) | Positive / negative CI |
|---|---:|---:|
| neural_vs_full | -10.8544 to 3.4851 | 0 / 16 |
| floor_vs_full | -1.0297 to 2.4841 | 15 / 9 |

## Risk Reliability

Predicted positive-harm ratio is not a calibrated safety certificate. Realized positive harm and net easy degradation are different quantities.

| Policy | Supported selected locality/views | Above 2% realized harm ratio | Underpredicted | Realized ratio range | Predicted ratio range |
|---|---:|---:|---:|---:|---:|
| global | 288 | 116 | 236 | 0.0000 to 0.1430 | 0.0010 to 0.0128 |
| producer | 288 | 102 | 233 | 0.0000 to 0.1525 | 0.0011 to 0.0127 |
| placebo | 288 | 110 | 237 | 0.0001 to 0.1454 | 0.0010 to 0.0126 |
| wrong_tag | 288 | 109 | 235 | 0.0001 to 0.1679 | 0.0008 to 0.0127 |
| legacy | 288 | 103 | 232 | 0.0003 to 0.1163 | 0.0009 to 0.0127 |

## Post-Readout Easy-Error Attribution

Exploratory arithmetic on frozen metrics, not a newly selected policy. For each locality, total degradation versus CV equals branch-floor degradation plus the controller increment, with one common CV denominator.

| Policy | Locality/view violations | Already-bad floor locality/views | Violations newly introduced by controller | Violations despite controller improvement |
|---|---:|---:|---:|---:|
| global | 7 | 12 | 0 | 6 |
| producer | 7 | 12 | 0 | 6 |
| placebo | 7 | 12 | 0 | 6 |
| wrong_tag | 7 | 12 | 0 | 6 |
| legacy | 7 | 12 | 0 | 6 |

Every event, branch, fold, seed and locality is retained in groups/*.json and *_localities.csv, including tail errors.
Partial trajectories only contribute observed labels; missing labels are not zero error. Complete-window ADE and endpoint FDE use their actual support.
EuropeanSquares released detector tracks, image-pixel obs8/pred12, raw annotation stride12. Not historical raw-t50, metric, seconds, human gold, true3D or foundation evidence.
Opened sources remain development. Independent selection/calibration/confirmation remain closed. No deployment change, Stage5C or SMC.
