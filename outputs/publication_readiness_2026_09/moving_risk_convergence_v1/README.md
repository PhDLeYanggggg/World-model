# Fixed-Budget Continuation: Optimization And Transfer

I continued the eight saved risk heads while keeping architecture, features,
optimizer settings, source separation, targets and decision rules fixed.
Every fold received at least 4,096 additional updates, rounded to full epochs.
Only the final checkpoint was evaluated after all eight fits and refits completed.

The registered information screen passes **0/3** contexts.
This screen is not the original full-policy safety estimator. No deployment change.

## Decision Results

| Context | Model | Risk-MSE reduction % | Decision-cost reduction % | Admitted | Harmful |
|---|---|---:|---:|---:|---:|
| critic0 | shared64 | +0.000000 | +0.000000 | 1 | 0 |
| critic0 | track32 | -0.426663 | -4.481672 | 979 | 107 |
| critic0 | short_plain | -28.280399 | -8.840174 | 549 | 72 |
| critic0 | short_transport | -42.675320 | -6.914305 | 755 | 94 |
| critic0 | long_plain | -257.937251 | -4.419872 | 2208 | 208 |
| critic0 | long_transport | -257.916490 | -4.719181 | 2278 | 214 |
| critic1 | shared64 | +0.000000 | +0.000000 | 10 | 1 |
| critic1 | track32 | -0.198968 | -121.691096 | 1762 | 723 |
| critic1 | short_plain | -20.095533 | -65.146776 | 788 | 320 |
| critic1 | short_transport | -18.758902 | -54.521863 | 956 | 353 |
| critic1 | long_plain | -26.258786 | -68.970280 | 1267 | 494 |
| critic1 | long_transport | -26.393744 | -68.647572 | 1277 | 498 |
| critic2 | shared64 | +0.000000 | +0.000000 | 0 | 0 |
| critic2 | track32 | +33.419794 | -2.568241 | 2165 | 521 |
| critic2 | short_plain | +33.427388 | -6.467748 | 4606 | 1165 |
| critic2 | short_transport | +32.191628 | -5.517107 | 4785 | 1288 |
| critic2 | long_plain | +14.060952 | -4.585305 | 3800 | 942 |
| critic2 | long_transport | +11.915158 | -6.692049 | 4656 | 1219 |

Positive reductions are better; shared64 is the reference. Costs are source-macro
known-proposal decision costs, not trajectory ADE/FDE improvements. Short and
long mean the 32-epoch parent and fixed-budget continuation of the same network.

## Fitting Versus Source-Held Error

| Excluded source | Fit rows | Added updates | Final epochs | Fit loss, before | Fit loss, after | Held plain MSE, before | Held plain MSE, after |
|---|---:|---:|---:|---:|---:|---:|---:|
| moving_source_component_048 | 3998 | 4096 | 544 | 0.362329 | 0.025690 | 1.133738 | 1.618858 |
| moving_source_component_067 | 1769 | 4096 | 1056 | 0.483918 | 0.004197 | 1.720374 | 6.344891 |
| moving_source_component_053 | 3043 | 4098 | 715 | 0.469824 | 0.018839 | 1.309648 | 0.497073 |
| moving_source_component_065 | 3671 | 4096 | 544 | 0.500936 | 0.034209 | 1.514367 | 2.819417 |
| moving_source_component_076 | 716 | 4096 | 2080 | 0.525897 | 0.000365 | 7.738072 | 7.787639 |
| moving_source_component_059 | 29807 | 4130 | 102 | 0.381098 | 0.300194 | 2.131963 | 2.955937 |
| moving_source_component_060 | 27830 | 4125 | 107 | 0.402854 | 0.320348 | 0.643288 | 0.991421 |
| moving_source_component_073 | 19521 | 4134 | 138 | 0.252503 | 0.110682 | 1.220390 | 1.210644 |

Additional primary updates: 32,871. Fitting loss improves in 8/8 folds; source-held plain MSE improves in 2/8.
The registered last-quarter fitting plateau diagnostic holds in 0/8 folds.
Neither completing the budget nor this operational diagnostic proves convergence.
The full loss trace is retained; no checkpoint was chosen on held outcomes.

## Evidence Boundaries

- fresh_run: continued optimizer updates, full repeated continuations and final development readout.
- cached_verified: parent networks, causal features, reference IDs, shared64 and track32 controls.
- not_run: new trajectory training, new reserved evaluation, three-seed main result, deployment.
- All 48,061 known proposals remain; 1,274 unknown outcomes are not made safe.
- Original 2% all/easy harm, easy-net and support rules remain. Empty support is NA.
- No identity exemption; the original 24-group primary remains NA, not passed.
- Source-held own-B has already been exposed for development; replay is not independence.
- Native obs8/pred12 image-coordinate silver labels, not metric or seconds-level.
- No true3D, foundation, Stage5C, SMC or submission-ready claim.

[Protocol](protocol.md) | [Loss trace](loss_trace.csv) | [Failure analysis](failure_analysis.md)
| [Reproduction](operation_zh.md) | [Verification](verification_report.md)
