# Float32 Odds-Ratio Backward Failure: Preserved Precursor

Status: fresh_run training failed;37/54 finals completed, no training freeze,
no fit/held model readout. PID40764 exited1. This is neither a completed
experiment nor evidence that the scientific hypothesis failed.

Failure: `single2_seed43_controller0_dimensionless`, head seed17, joint_risk.
The last consistent atomic checkpoint is update1600. Loss was finite, but the
gradient-norm guard rejected a nonfinite backward update. A diagnostic fork of
that saved checkpoint reproduced the failure with autograd anomaly tracing:
`MulBackward0` returned NaN at the bounded harm odds update, line33 of the
registered model. The original checkpoint and37 final receipts were not changed.

The float32 ratio has finite forward values at zero/tiny envelopes, while its
backward intermediates can overflow before multiplication by zero creates NaN.
For the minimal zero-envelope case with positive odds shift, the numerator
gradient is8.507059e37, its preceding product gradient becomes inf, and the harm
gradient becomes NaN. This is an arithmetic defect, not the historical OpenMP/SHM hang,
not exhausted memory and not evidence from a held/test metric.

Repair: a separately registered `european_isolated_harm_fp64_v1` computes only
the bounded ratio intermediates in float64 and returns float32 harm outputs.
The original float32 denominator floor is retained; no risk threshold, loss,
sample budget, data role or architecture is relaxed. Near-zero backward cases
must pass. Because numerical rounding changes optimization, all54 comparisons
restart from the original shared-utility anchors. No successful precursor fit
is pooled with repaired fits. Stage5C/SMC/deployment remain disabled/unchanged.

Private diagnostic fork:
`data/stage_cvpr2027_experiments/european_isolated_harm_v1/diagnostic/anomaly_joint.pt.gz`.
The bound original implementation and registration remain preserved in Git.
