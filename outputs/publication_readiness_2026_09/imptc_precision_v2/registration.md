# Fixed Canonical Precision Probe

Registered before execution. The previous unit-contract run found three EqMotion
normalized-output differences above1e-5 despite the same past represented in
different coordinate units. Tiny float32 boundary changes, not future errors,
identified the numerical mechanism. The original failed gate is retained.

Use the exact frozen58-query/754-window engineering selection and checkpoints
from imptc_input_contract_v1, with coordinate factors .01/1/100. Add one numerical
change: round shared canonical prefix coordinates to nine decimal places before
target-local normalization. This is finer than float32 nominal precision and
matches the already fixed1e-9 dimensionless neighbor tie precision. It is not a
new unit, source clock, resampling or interpolation. Do not tune the precision
based on future errors. Tiny motion may be removed; count such windows explicitly.

Run all fixed unit probes and report maximum normalized predictor/feature
differences under the unchanged1e-5 tolerance. Do not assert invariance outside
these finite inputs/units or claim lossless input reconstruction. No fitting,
forecast-error evaluation, source admission, deployment, Stage5C or SMC.
IMPTC covariates are already design-exposed; DroneCrowd stays closed.
The concurrently registered six-moment risk-head fit is a separate source-only
experiment with unchanged old prefix geometry, not this quantized adapter.
