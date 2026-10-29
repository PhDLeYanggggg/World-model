# Preserved Numerical-Assertion Interruption

PID64212 exited1 on29October2026 after12/72 fitting views and4/36 complete
probe groups. It failed at context single0_seed43_controller1_dimensionless,
headseed43, self arm, next-batch7. No held inference or persistent update.
The existing receipts and registration are preserved; this run is incomplete.

The original per-element gradient-sum check rejected2/12,322 coordinates.
A same-batch numerical reproduction gives float32 component-relative norm
error1.68805297e-7, versus float64 error2.88992839e-16. The largest absolute
error is0.001484 on a gradient of about7,668; the assertion's failing small
coordinates are not the largest absolute-error coordinates.

This is a reduction-precision checking defect, not evidence that the training
objective has missing terms. In v2, the native float32 vector check uses a
component-magnitude norm bound of32 float32 eps, and every batch also gets
a separately reconstructed double-precision decomposition check at1e-10.
The actual models, native gradients, AdamW states, steps, samples, outcomes,
scientific thresholds and loss weights remain unchanged. All72 views and288
probe batches will be rerun, with old completed native receipts cross-checked.
No v1 receipt is overwritten or relabeled complete.

Reproduction: `scripts.diagnose_m3w_query_additivity`, at its original recorded
script hash and pre-v2 code revision. It is not a scientific model comparison.
