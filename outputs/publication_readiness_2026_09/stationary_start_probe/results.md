# Stationary-Start Source Audit and Fit-Only Probes

Fresh source replay and 48 fitted classifiers; no development, calibration or confirmation labels opened.
The parent eight-observed/twelve-predicted-step task and primary forecasting metric are unchanged.

## Source Support

| Fit recording | All fit windows | Stationary windows | Changed future | Agents | Stationary runs | Runs with change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eth_eth | 2614 | 81 | 59 | 5 | 11 | 11 |
| eth_hotel | 1197 | 284 | 129 | 26 | 34 | 25 |
| ucy_zara01 | 2234 | 0 | 0 | 0 | 0 | 0 |
| ucy_zara02 | 5741 | 0 | 0 | 0 | 0 | 0 |
| ucy_zara03 | 180 | 0 | 0 | 0 | 0 | 0 |

Canonical source positions exactly reproduce every cached row in the five fit recordings.
No malformed source row or history gap occurs here. This proves cache lineage, not true physical stillness
or causal annotation construction. Any future coordinate change is a diagnostic label, not an intent label.
The 365 windows are 31 agents and 45 stopped runs. Repeated seeds and overlapping windows are not independent sites.
Zara has no exact-stationary windows; its stationary held-fold comparison is not_run, not a success.

## Fixed Models and Single-Factor Repair

Initial ordered context uses 19 geometry or 83 geometry/motion features. The adaptive repair uses
six/13 permutation-invariant summaries. Same rows, folds, labels, seeds, logistic C=1 and ExtraTrees
256/min-leaf10; no threshold search. Feature scaling is observed-query/fit-only. Every fitted setting is shown.
Positive Brier lift means lower probability error than the opposite-scene training-only prior.
The prior rates differ: Hotel->ETH 45.45% versus held 72.84%; ETH->Hotel 72.29% versus held 45.42%.

| Features | Held | Model | Context | Mean AUC | Brier lift | Run-balanced lift | Agent-balanced lift | Positive seeds |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ordered | ETH | logistic | geometry | 0.5740 | -0.06521 | -0.05123 | -0.08248 | 0/3 |
| ordered | ETH | extra_trees | geometry | 0.5588 | -0.01393 | +0.02276 | +0.00150 | 0/3 |
| ordered | ETH | logistic | geometry_motion | 0.4461 | -0.15762 | -0.10901 | -0.16146 | 0/3 |
| ordered | ETH | extra_trees | geometry_motion | 0.4417 | -0.06208 | -0.03870 | -0.03104 | 0/3 |
| ordered | Hotel | logistic | geometry | 0.5683 | -0.07024 | +0.00402 | +0.01297 | 0/3 |
| ordered | Hotel | extra_trees | geometry | 0.4913 | -0.03023 | -0.00208 | +0.02142 | 0/3 |
| ordered | Hotel | logistic | geometry_motion | 0.6005 | -0.13477 | -0.06668 | -0.07656 | 0/3 |
| ordered | Hotel | extra_trees | geometry_motion | 0.4301 | -0.10018 | -0.04545 | -0.04085 | 0/3 |
| pooled | ETH | logistic | geometry | 0.4337 | -0.11301 | -0.12204 | -0.13015 | 0/3 |
| pooled | ETH | extra_trees | geometry | 0.6949 | +0.02513 | +0.02012 | +0.04701 | 3/3 |
| pooled | ETH | logistic | geometry_motion | 0.2766 | -0.20060 | -0.20459 | -0.20567 | 0/3 |
| pooled | ETH | extra_trees | geometry_motion | 0.6692 | -0.03195 | -0.04570 | -0.00882 | 0/3 |
| pooled | Hotel | logistic | geometry | 0.5188 | -0.06473 | -0.01502 | -0.00675 | 0/3 |
| pooled | Hotel | extra_trees | geometry | 0.4895 | -0.01943 | -0.00285 | +0.00494 | 0/3 |
| pooled | Hotel | logistic | geometry_motion | 0.5041 | -0.10187 | -0.14137 | -0.06482 | 0/3 |
| pooled | Hotel | extra_trees | geometry_motion | 0.5003 | -0.00820 | +0.00325 | +0.01359 | 0/3 |

## Evidence Boundary

Saved model predictions reproduce all 48 reported Brier scores within 1e-12. Run/agent reweighting is descriptive;
it does not establish independence or replace the registered forecasting metric. No confidence interval or
statistical significance claim is made for the two exposed fit sites. Deterministic logistic seeds are identical,
not three independent replications. The pooling follow-up was chosen after seeing the first diagnostic and
is explicitly adaptive. It is not a final-test improvement or a new deployment.

Scene image cues, future direction prediction and a start-aware neural trajectory head were not run.
No metric/seconds, true-3D, foundation, Stage5C or SMC claim. The initial serialization failure and repair
are preserved in extraction_failure.md. Full original/follow-up results are in metrics.json and pooled_metrics.json.
