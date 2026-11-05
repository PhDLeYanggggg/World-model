# Signed-Cost Neural Admission: Completed Comparison

I changed the training objective, not the evaluation contract. Six real
neural continuations compare magnitude-weighted logistic sign learning
with MSE from identical warm weights. Unknown outcomes retain their
original bound-aware MSE. Neither output is a calibrated safety bound.

## Held Development Results

| Model | Known ADE gain over floor % | Hard gain % | Intervention % | Full risk/support |
|---|---:|---:|---:|---:|
| readout_veto | 4.3233 | 5.3547 | 22.0115 | 0/17 |
| kinematic_veto | 3.9418 | 4.8582 | 17.3155 | 0/17 |
| continued_mse_veto | 3.6006 | 4.6257 | 15.0144 | 0/17 |
| in_source_errors_veto | 3.2049 | 4.0374 | 13.9823 | 0/17 |
| held_source_errors_veto | 2.9544 | 4.0590 | 14.1132 | 0/17 |
| signed_cost_logistic_veto | 2.6928 | 3.5649 | 17.6705 | 1/17 |

Signed-cost minus MSE known contrast: **-0.9078 pp**.
Missing-label completion bounds: [-2.2534, 0.4379] pp.
These are not sampling confidence intervals.
Prospective expansion screen: **False**. It is not a safety certificate.

## Original Safety Accounting

| Arm | Known all violations | Known easy violations | Undefined selected easy | Worst defined easy-net % |
|---|---:|---:|---:|---:|
| continued_mse | 14 | 13 | 2 | 38.9046 |
| signed_cost_logistic | 11 | 14 | 1 | 26.2254 |

Full original24group primary remains NA, not passed. Missing supports
and empty same-count controls do not become safety passes. No identity
certificate or evaluation revision. Prior failures remain unchanged.

## Runtime And Reproducibility

| Phase | PID | Seconds | Peak RSS bytes |
|---|---:|---:|---:|
| pilot | 84463 | 36.32 | 2843328512 |
| train | 84628 | 517.02 | 7521599488 |
| evaluate | 85449 | 46.57 | 4479221760 |
| verify | 85536 | 45.66 | 4482891776 |

Nativearm64 Torch CPU4/interop1/workers0, no resource probing. Six fits
complete12,000primary updates; exact treatment refit adds2,000. Both
arms resume exactly and all three fresh MSE controls reproduce old weights.
All models freeze before V. New predictions/actions/metrics replay exactly.
Old control actions are cached_verified; their metrics are recomputed
fresh and match. New loss fits and evaluation are fresh_run.

## Limits

Seven exposed SOURCE units,17recording views, one producer seed. Many
overlapping windows do not add independent scene evidence. This is a
repeated development comparison, not independent confirmation. B has
only2/3/3source groups. Logistic logits are not expected-risk estimates;
unknown-bound MSE makes the complete objective a mixed surrogate.
No new trajectory dynamics, JEPA, held H or reserved external readout.
Nativeobs8/pred12 imagepixels, not verifiedseconds/metric/true3D/foundation.
No deployment change or submission-readiness upgrade. Stage5C/SMCoff.

[Protocol](protocol.md) | [Fitting audit](full_fitting_objectives.json) | [Exact replay](verification.json)
