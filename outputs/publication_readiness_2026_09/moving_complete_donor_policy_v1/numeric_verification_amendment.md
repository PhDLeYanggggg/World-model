# Float-Parity Repair Before Full-Policy Readout

Initial definition67a8d76d and registration.json are retained. Inference PID69915
exits1 after the first context succeeds. In the second context, one of5,998
score entries differs1.06280145e-9, just outside the original1e-9absolute check.
The raw120features already match exactly. This occurs at the old final-batch
tail after unknown rows change batch shape. No full-policy outcome readout
has run. Do not retrain, use future-validity masks to select batches, or alter
the risk thresholds to address a numerical comparison.

Version2changes only the verification contract: compare scores in saved
training-target-RMS units using rtol/atol64times float32epsilon, the fixed
largest hidden width. This is an engineering tolerance, not a formal numerical
error theorem or risk allowance. Record maximum differences and all old scalar
tolerance violations. Every channel's zero-threshold decision must match, and
the final mean/unanimity choices must separately match exactly on known rows.
Features and forest scores still match exactly. Any decision mismatch halts.

Model weights, data, cohort, inference math/batches, proposal choices, threshold0,
original2%risk/support rules and evaluation metrics do not change. In particular,
this does not introduce an identity-fallback certificate. Version1failure and
its first-context cache stay intact. Version2inference uses separate cache and
receipt directories; the unchanged packed population is reused with hashes.
The new registration binds the original definition and this amendment before
resuming inference or opening the complete-policy metrics.
