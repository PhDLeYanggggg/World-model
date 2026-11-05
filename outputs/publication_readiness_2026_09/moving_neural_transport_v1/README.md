# Neural Risk Transport: Trained, Not Promoted

I trained a source-only risk regressor and used its predicted difference
between a query and its fixed references to adjust the reference outcomes.
The same fitted network also supplies a plain prediction control. A ridge
model separates linear adjustment from nonlinear learning. The features,
reference IDs, source roles and original risk rules are unchanged.

**Transport helps relative to the plain network, but not enough to beat
the existing forest or repair safe intervention.**

| Context | Model | Risk-MSE reduction % | Decision-cost reduction % | Admitted | Harmful |
|---|---|---:|---:|---:|---:|
| critic0 | shared64 | +0.000000 | +0.000000 | 1 | 0 |
| critic0 | track32 | -0.426663 | -4.481672 | 979 | 107 |
| critic0 | ridge_plain | -16.468840 | -0.922995 | 494 | 54 |
| critic0 | ridge_transport | -33.408988 | -1.489281 | 510 | 53 |
| critic0 | neural_plain | -28.280399 | -8.840174 | 549 | 72 |
| critic0 | neural_transport | -42.675320 | -6.914305 | 755 | 94 |
| critic1 | shared64 | +0.000000 | +0.000000 | 10 | 1 |
| critic1 | track32 | -0.198968 | -121.691096 | 1762 | 723 |
| critic1 | ridge_plain | -15.596289 | -0.119843 | 410 | 89 |
| critic1 | ridge_transport | -13.822722 | +0.325424 | 434 | 92 |
| critic1 | neural_plain | -20.095533 | -65.146776 | 788 | 320 |
| critic1 | neural_transport | -18.758902 | -54.521863 | 956 | 353 |
| critic2 | shared64 | +0.000000 | +0.000000 | 0 | 0 |
| critic2 | track32 | +33.419794 | -2.568241 | 2165 | 521 |
| critic2 | ridge_plain | +2.070491 | -20.828380 | 6467 | 1729 |
| critic2 | ridge_transport | +4.098059 | -18.957826 | 6923 | 1847 |
| critic2 | neural_plain | +33.427388 | -6.467748 | 4606 | 1165 |
| critic2 | neural_transport | +32.191628 | -5.517107 | 4785 | 1288 |

Positive reductions are better; shared64 is the reference. These are
source-macro known-proposal decision costs, not ADE/FDE gains or the
full-policy scene-level safety estimator.
The registered primary information screen passes **0/3** contexts.
Secondary arms cannot retrospectively replace the primary. No deployment change.

## Actual Training

Eight MLPs (9,890 parameters each), eight ridge fits and 5,792 primary optimizer updates complete.
Every donor-fitting row receives 32 epochs. The real pilot verifies exact
one-epoch resume versus uninterrupted two-epoch training. All eight full
neural and ridge fits then reproduce exactly. No NumPy fallback substitutes
for the Torch networks. No new trajectory/dynamics model is trained.

| Excluded source | Fit rows | Updates | First logged fit loss | Epoch 32 fit loss | Source-held plain neural MSE |
|---|---:|---:|---:|---:|---:|
| moving_source_component_048 | 3998 | 256 | 0.972873 | 0.362329 | 1.133738 |
| moving_source_component_067 | 1769 | 128 | 0.982397 | 0.483918 | 1.720374 |
| moving_source_component_053 | 3043 | 192 | 0.961695 | 0.469824 | 1.309648 |
| moving_source_component_065 | 3671 | 256 | 0.968626 | 0.500936 | 1.514367 |
| moving_source_component_076 | 716 | 64 | 0.980088 | 0.525897 | 7.738072 |
| moving_source_component_059 | 29807 | 1888 | 0.928734 | 0.381098 | 2.131963 |
| moving_source_component_060 | 27830 | 1760 | 0.949998 | 0.402854 | 0.643288 |
| moving_source_component_073 | 19521 | 1248 | 0.966941 | 0.252503 | 1.220390 |

Loss is weighted two-channel MSE after donor-only target-RMS normalization.
The first log is after one optimizer update, not an untrained-model score.
Epoch 32 is the fixed final checkpoint, not validation-selected. Loss still
decreases late in several folds; convergence is not established. The large
fit-versus-source-held gap also precludes assuming more epochs will fix transfer.
[Complete loss trace](loss_trace.csv).

## Evidence Boundaries

- fresh_run: eight actual Torch risk fits, eight ridge controls, full refits and inference.
- cached_verified: causal features, labels, fixed reference IDs, track32 and shared64 controls.
- not_run: new neural trajectories, full-policy deployment, nested-C/H, independent confirmation.
- All 48,061 known proposals remain; 1,274 unknown proposals are not fabricated safe.
- References are source-excluded but their residuals are in-sample for the fitted donor model.
- Residual transport is an established type of regression correction, not a novelty claim.
- Original 2% all/easy harm, easy-net and support rules remain; empty support stays NA.
- No identity exemption, original 24-group primary pass, reserved readout or submission readiness.
- Native obs8/pred12 image-coordinate silver data, not metric/seconds/true3D/foundation.
- Stage5C/SMC remain disabled.

[Protocol](protocol.md) | [Failure analysis](failure_analysis.md) | [Reproduction](operation_zh.md)
| [Related work and paper note](paper_addendum.md)
