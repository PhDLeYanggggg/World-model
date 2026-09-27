# Scientific and Operational Gates

Source: `aggregate_metrics.json`; protocol frozen before training and readout.
This is a cost-estimation mechanism study, not the historical stage gate count.

| Gate | Status | Evidence |
|---|---|---|
| Fixed training completed | pass | 1008 heads,2016000 updates,zero unknown-label draws |
| Readout execution completed | pass | 432 fits,144 views,1728 direct MSE checks |
| Primary cost | fail | Full true vs scaled-cost and vs raw-true each2/6 positive CIs,required6/6 |
| Harm/coverage guards | fail | True vs raw-true:4 negative coverage-log and1 negative top10 intervals |
| Auxiliary information | fail | Full true vs scaled-shuffled2/6 positive CIs,required6/6 |
| Advance | fail | Conjunction of primary,guards,information fails |
| Exact replay/scoped tests | see verification.json | Only a completed seal establishes replay status |
| Independent confirmation | not_run | Independent selection/calibration/confirmation stay closed |
| New trajectory/deployment lift | not_run | Frozen forecasts,no intervention policy evaluation |
| Submission readiness | false | Independent evidence and main-method comparisons remain missing |
| Stage5C execution | false | Prohibited |
| SMC | false | Prohibited |

Engineering checks do not offset failed scientific gates. The number of passed
checks is not a model-quality score. Bootstrap is descriptive source evidence,
not confirmation or a formal safety guarantee. Obs8/pred12 native steps,
detector pixels; no metric/seconds,true3D,foundation or human-gold claim.
