# Manuscript Evidence Addendum

## Controlled Diagnostic

We separate forecaster fitting, controller fitting, incremental-head fitting and
readout at locality level. A fixed learned damping policy provides identical
fallback producers for the last two roles. This avoids interpreting a change in
floor-training exposure as an effect of the incremental cost target.

With coordinate-unit-repaired forecast banks, the floor/neural oracle admits
21.25% ADE improvement, while a fixed linear moment screen realizes0.52%
[0.37%,0.69%]. Positive-only selection realizes9.11% but harms easy cases.
Reference-consistent target fitting does not improve its matched CV-target
control (-0.0169%,95% interval[-0.0218%,-0.0115%]). The screened policy's
selected positive-harm ratio is4.91%, exceeding the2% target despite every
observed view preserving easy mean error. Thus net preservation does not imply
conditional harm control, and reference alignment alone is insufficient.

All reported uncertainty resamples twelve development-exposed locality means,
averaging dependent seeds and producer contexts first. This is not an
independent test or simultaneous confidence statement. Future-cost oracles are
diagnostics only; no new deployable model or broad world-model claim follows.

The research contribution still needs a method that controls conditional harm
without discarding nearly all incremental opportunity, plus held-scene
calibration, independent confirmation and matched scene-joint controls.
