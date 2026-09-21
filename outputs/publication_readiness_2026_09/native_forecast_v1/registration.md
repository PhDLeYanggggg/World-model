# Full-Population Native-Loss Comparison

Registered 2026-09-21 before new training or held-source scoring. This is
exploratory protocol development after the disclosed native-metric amendment,
not independent confirmation or a replacement for old negative results.

## Hypothesis and Fixed Matrix

Does correcting per-query error weighting allow an existing causal predictor to
improve ordinary motion without discarding static or unsupported-label queries?
Compare two losses on the same existing Transformer, four source-site exclusions,
three seeds: 24 fits, 4,000 updates each, 96,000 updates total. This isolates the
loss/weighting change, not a new architecture or full multimodal contribution.
No outcome-based winner or checkpoint selection. Every final endpoint is reported.

Reuse the 64-wide, two-layer, four-head past-context Transformer with the existing
observed-context conditioner and ego-motion-bounded baseline-relative output.
Both arms initialize exactly at causal CV. The bound is past-only. It prevents
nonzero forecasts after exactly static ego history, including real static starts;
those misses stay in the loss and scores. This is an explicit action-class
limitation, not a claim that motion conditioning solves intent prediction.

Use all 175,756 admitted source queries from coupa/deathCircle/gates/hyang, original
SDD train only. Each fit excludes one physical site and fits the other three.
Bookstore and all main/external/original validation/test roles remain closed.
Preserve masks and all past-eligible rows, including absent/partial supervision.
No fitted parent, teacher, goal cluster, source ID or future support mask enters
the prediction payload. Context includes eight past ego steps, aligned observed
neighbor histories and the causal CV rollout. It excludes images in both arms.

## Loss and Sampling

Let e_i be masked 12-step ADE in the old past-normalized coordinate system and
s_i its causal native scale. The two losses use l_i=e_i or l_i=s_i*e_i.
Within each training scene, divide mean supported l_i by the corresponding
training CV mean. Average these relative risks equally over training scenes.
This aligns the native arm with within-scene relative ADE comparison while
keeping target scale distinct from observed input conditioning.

Sample training scenes uniformly, then all indexed rows uniformly within scene.
Unknown-supervision rows have zero loss, not fabricated targets. Multiply each
supported loss by N_indexed/N_supported for its training scene, so that this
sampler has exactly the intended supported-scene mean objective in expectation.
Tests must verify that correction analytically. Both arms use identical seeds,
batch IDs, optimizer, cosine schedule and budgets. Input conditioning, loss
normalizers and support corrections use only the training complement. No held
outcomes affect a fit or its preprocessing.

## Execution and Recovery

Native arm64 .venv-pytorch, CPU four threads / interop one / workers zero, no
resource probing. First run 100 updates of one registered trial to measure real
training throughput, checkpoint and verify resume. The pilot is part of the
fixed budget; do not score its held scene or choose a model from it. Then finish
all endpoints. Use local execution when the measured cost is reasonable;
otherwise inspect CREATE resources and existing jobs before migration.

Atomic checkpoints preserve model, optimizer, RNG, sampled-row counts, config
and input/source identity. PID/step/loss/elapsed heartbeats every 50 updates and
checkpoints every 200. Resume validates dependencies and settings. An interruption
never authorizes deleting evidence or skipping the remaining budget. Final
source readout starts only when all 24 receipts pass completion checks.

## Fixed Evaluation and Safety Boundaries

Apply the adopted native metrics to paired supported masks, with complete-future
sensitivity and the full old metric as diagnostics. Compare both loss arms,
causal CV and the other-source-sites-selected fixed baseline. Keep all seeds,
per-scene errors, worst-scene result and tail error. Use 3,000 physical-scene
resamples, explicitly conditional on four explored sites and shared training.

Report exact-zero-CV easy absolute harm separately; its percentage is undefined.
Report stationary-history, moving-history and training-complement-q75 native-CV
hard slices as descriptive diagnostics, not a newly calibrated deployment gate.
Do not silently transplant old-scale easy/failure thresholds. The overall <=2%
easy-safety objective remains, but no deployment is permitted until a native-scale
risk definition and independent calibration are separately registered and tested.

No threshold search, risk-head training, scene-policy selection, independent
confirmation, Stage5C or SMC. No meter/seconds/true-3D/foundation claim. A positive
loss contrast is not automatically a deployable world-model or paper contribution.
