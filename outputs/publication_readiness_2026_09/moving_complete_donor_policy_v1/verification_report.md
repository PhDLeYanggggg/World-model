# Verification Of The Complete-Policy Result

fresh_run: complete reconstruction, frozen inference, metrics and serial replay.
cached_verified: original models, forecasts, splits and known-row predictions.
not_run: new training, independent C/H readout, deployment or full historical tests.

## Scientific Result

All original full-policy risk/support checks are retained. Mean and unanimity
fail, with only 1/24 and 2/24 supported recording views. Positive trajectory
contrasts do not override these failures. Original 24-group main remains NA.

## Runtime And Replay

Native arm64 Torch, CPU 4 threads, interop 1, workers 0. Prepare PID69798,
infer PID70312 and evaluate PID70430 all terminate with exit 0. The initial
v1 infer PID69915 exits 1 on numerical parity; the failure is not removed.

Final replay PID71145 terminates with exit 0 in 13.616047709 s, peak RSS
4,048,846,848 bytes. It reruns all three full causal contexts, compares every
cached array exactly, recomputes all metrics exactly and verifies every public
file remains unchanged during replay. No fitting occurs. This is software
reproducibility, not independent scientific confirmation.

Against historical known-row caches, all 120-dimensional features and all
choices are exact. Only 8/24 raw risk-score arrays are bitwise equal after full
population batching; maximum absolute score roundoff is 4.908907e-9. All
zero-threshold channel and ensemble decisions match. The engineering-only
amendment is explicit and does not relax scientific risk limits.

## Focused Checks

19 tests pass in 1.73 s: full-policy veto/support, causal packet equivalence,
float parity with sign protection, failure reporting, seed consensus, donor
features and same-count utility. The full historical test suite was not rerun.
The local gain/risk figure was visually inspected. It separates trajectory
benefit from the unpassed 24-view safety requirement.

The compiled paper is unchanged. Its recorded SHA256 is
`d9babe3ddbac858bcc1ac30e3ba98b332bed625c78ff9231d1d42a3e30b11772`.
No new CREATE job or reserved data readout. Only code and small text/JSON/CSV
artifacts are intended for Git; private arrays, weights and images remain local.
