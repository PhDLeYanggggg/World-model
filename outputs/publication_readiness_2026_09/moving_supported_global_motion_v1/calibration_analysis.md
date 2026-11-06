# Held-Source Cost And Observation Diagnostics

Posthoc,already-exposed own-B only. No new fit,threshold or calibration.
Conditional MSE uses the common causal history scale,not each arm's
different fitting RMS. Values average equally across8sources and3risk seeds.

| Arm | B MSE | H MSE | R MSE | ER MSE | EH MSE | Observability Brier | Observability ECE | Context clipped fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own_local | 0.004656 | 0.002364 | 0.115037 | 0.000993 | 0.000413 | 0.083092 | 0.094877 | 0.003437 |
| own_global | 0.004784 | 0.004021 | 0.194342 | 0.008340 | 0.001588 | 0.068581 | 0.072510 | 0.003437 |
| augmented_local | 0.003665 | 0.001347 | 0.103832 | 0.000563 | 0.000252 | 0.043727 | 0.038634 | 0.000747 |
| augmented_global | 0.003911 | 0.001398 | 0.111094 | 0.000525 | 0.000274 | 0.047694 | 0.049338 | 0.000747 |

Future observability is annotation support,not trajectory-failure risk.
It is evaluated once per agent-window,not six times per amplitude.
A local arm's clipping statistic describes its masked context,not an input
it used. Easy-credit mass on realized nonEasy rows is an attribution,not
by itself proof of an Easy classification or calibration failure.
Selected credit/harm totals are in calibration_analysis.json; they do not
replace the original per-record2%and unknown-completion contract.

## Local Replication Control

18/24augmented-local models exactly recover
the original role-separated parent weights. Masked context is checked as zero;
normalization and samplers are recorded per pair. This is a consistency
check,not independent evidence or a newly discovered method.
