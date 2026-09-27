# Data and Model Scope

Source: previously admitted European source-training recordings, with a sealed
causal geometry cache. 318,969 target histories, 163 recordings, 12 localities.
Detector-derived trajectories are silver, not human gold. Training/locality
roles and all preprocessing are inherited from the validated source lineage.
No reserved model-selection, calibration or confirmation data is read.

Prediction: 8 observations, 12 requested future steps, raw-frame stride12.
Geometry is image-local. No seconds, homography/metric, physical safety or
true3D claim follows. Historical t+50 results are a different task and are not
reconfirmed by this experiment. Overlapping windows are not independent units.

Inputs: ego past, up to8 current-visible neighbors with observed history masks,
causal baseline rollout and requested horizon. No future positions, future
valid masks, future goals, central velocity or test statistics enter inference.
No scene images or learned goal maps are added by this topology comparison.

Model: width64, four heads, one within-agent temporal layer, masked pooling,
one between-agent layer, future query decoding, motion-bounded deterministic
correction. Same 88,514 parameter shapes and initialization as the flat bank.
Agent ordering remains equivariant; shared temporal histories retain supplied
track association. Cross-time track association may still be detector-noisy.
The encoder's use of all observed history is causal at prediction time, not
an autoregressive future rollout. Single-target predictions are not jointly
constrained scene futures.

Nine fits use three fixed producer folds and seeds17/29/43, 4,000 updates each.
Each producer fits four source localities and predicts the other eight. Each
held locality is excluded from its entire forecast/preprocessing chain. All
source-locality outcomes remain development-exposed; this is exploratory.

Native arm64 Torch CPU4/interop1, zero workers. Atomic checkpoints retain
optimizer, sampler and Torch RNG state. Independent calibration, deployment,
Stage5C and SMC are not performed. Raw data, caches and checkpoints stay private.
