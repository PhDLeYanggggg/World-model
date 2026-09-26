# Frozen Harm Readout Model Card

## Material Passport

Source-development risk-head experiment, not a new trajectory predictor or
deployment policy. Frozen encoders come from cached_verified mean and fractional
heads trained outside the held locality. All 288 new readouts are fresh_run
and complete. Neither registered primary comparison passed. This model is
diagnostic only; no new policy or deployment is authorized by its results.

## Architecture and Objective

The frozen risk head maps 383 causal features to 64 GELU hidden features. This
is not a newly trained JEPA/Transformer latent. Each hidden representation
receives the same linear 64-to-2 readout, 130 trainable parameters. Outputs are
H=e*sigmoid(a), H_easy=H*sigmoid(b), with the causal forecast-disagreement e.
D_all and D_easy copy the original mean control bit-for-bit. Two-harm squared
loss uses training-only RMS normalizers. No model output is used as a certified
probability or safety certificate.

Both arms share initialization, draw sequence, optimizer settings and 2,000 updates.
Three training localities determine normalization and easy definitions. No
future target enters hidden-feature extraction or inference. Future error is
a label for loss/evaluation only. Unknown labels are excluded from fitting.
The encoder is frozen and reference moments cannot change.

## Decision and Limits

Keep all source assignments and all three seeds. Improvement over the matched
readout alone is insufficient: the original-mean magnitude comparison must
also pass. No threshold scan, C policy evaluation or deployment this round.
Original feature training and this extra readout training are distinct compute
costs. A probe improvement would not prove a new world-model architecture.

Eight observed / twelve predicted annotation steps, raw stride 12, detector-derived image pixels.
No human-gold, metric/seconds, physical safety, true 3D or foundation claim.
Reserved calibration and confirmation stay closed. Stage5C and SMC remain off.
