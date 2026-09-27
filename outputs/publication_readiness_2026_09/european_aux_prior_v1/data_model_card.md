# Auxiliary Prior Repair: Data and Model Card

## Role and Lineage
Exposed European source-development views,not independent selection or test.
Obs8/pred12 native annotation steps;detector image pixels,not meters or seconds.
Existing383-dimensional causal forecast/context features and frozen forecasts
are reused with original row IDs,preprocessing hashes and nested risk labels.
Six producer/controller assignments,three seeds,two feature/forecast families,
four outer localities give144 dependent views. Full versus motion-only changes
forecast populations,so it is not a pure matched feature ablation.

Each head fits on the three non-outer localities. Inner producer-relative cap
events come from locality-excluded predictions. The outer outcome is unavailable
to fitting functions. Original target/feature hashes and sampled-known-row
support are verified. Preprocessing and initial event priors use fitting data
only. Shared containers are not filesystem-blind. There is no claim that these
historically exposed source-held localities constitute independent confirmation.

## Model and Intervention
An existing24901-parameter GELU64 cost estimator outputs bounded expected
denominator/harm moments plus a scalar cap-event logit. The experiment changes
only that logit's initial bias to the weighted finite fitting-event prevalence.
No new forecast model,scene encoder or dynamics head is trained. The predictor
does not receive a future endpoint,oracle label or future target as input.
Future-derived labels are only supervision/readout. Event probability never
multiplies the expected-harm prediction.

Two new matched arms use true and within-locality-shuffled cap-event labels.
Original cost-only and old true/shuffled arms are cached_verified controls,
not fresh training. New heads train2000 updates,AdamW,site-balanced256 batches,
four-cost normalized MSE plus coefficient1 binary cross-entropy on finite event
rows. Known zero-envelope rows still contribute to cost loss. Unknown cost
rows are never sampled. No hyperparameter search or checkpoint selection.

## Safety and Claim Boundaries
This model is an experimental error-cost estimator,not a calibrated deployment
policy or physical-safety system. Intervals resample only four localities and
are descriptive without multiplicity correction. A positive fitting or
source-held cost result is not evidence of ADE/FDE improvement or independent
generalization. Deployment remains unchanged. Independent roles stay closed.

No human-gold,true3D,foundation,metric or seconds-level claims. Stage5C and SMC
remain off. Training weights,row predictions,features and raw data stay out of
Git. Reproduction requires the hash-matched private source assets and native
Torch environment described in operations.md.
