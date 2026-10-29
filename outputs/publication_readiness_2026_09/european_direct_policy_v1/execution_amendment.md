# Execution Amendment: Prior Array Fingerprint Codec

The first real pilot, PID93772, exited1 before any optimizer update or model
checkpoint. Original input packets streamed successfully. Comparing the frozen
original action hash failed because the caller used the forecasting model's
array fingerprint, while the prior readout serialized arrays using the analogue
fingerprint (dtype + shape + bytes). A two-element Boolean array reproduces the
different digests even for identical content. The original prediction function
is the same callable; this is not evidence of a new data split or changed policy.

Execution revision2 uses the prior artifact's own fingerprint function for both
prior action and selected-moment checks. A regression constructs receipts using
that original codec and checks the same boundary. No prediction, feature, loss,
seed, sampling rule, threshold or evaluation criterion changes. The original
registration and commit13b87ab3 remain preserved. The revised execution binds
them and this amendment in execution_registration_v2.json before another pilot.

The failed attempt is not counted as training or scientific success. Exact real
control replay and resume must still pass in the revised pilot; no failed check
is bypassed. No independent data role was opened.
