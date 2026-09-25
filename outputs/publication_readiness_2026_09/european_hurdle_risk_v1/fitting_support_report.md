# Fitting-Only Event Support

`fresh_run`: 18 candidate/fold/seed training assemblies, two events and four
fitting localities each, producing144 aggregate slices. No new held-out outcome
readout or fitting was performed by this diagnostic. Source manifests, schemas,
OOF lineage and training targets were checked through the preceding verified
experiment. Rows and localities recur across seeds; do not sum them into an
independent sample size.

| Candidate | Event | Positive-harm rows per hypothetical256 local rows, range | Locality slices with no positive harm |
|---|---|---:|---:|
| Neural | All | 45.54-112.70 | 0/36 |
| Neural | Easy | 7.50-29.47 | 0/36 |
| Damping0.97 | All | 28.32-95.71 | 0/36 |
| Damping0.97 | Easy | 4.04-24.24 | 0/36 |

Actual draws are balanced among four localities per fit. The table expresses
local prevalence on a256-row scale; it is not a claim that each training batch
contains these counts. Exact per-batch positive counts appear in training logs.
Positive-harm fractions are nonzero and at most the causal rollout envelope.
Unknown future-cost labels remain unknown and are excluded from training.

The easy event uses the unchanged positive-CV-error threshold determined on
fitting sources. It is an evaluation/label definition, not an available inference
feature. In particular, it does not supply a future-defined easy label to the
model. A positive event-harm occurrence label means the candidate harmed a row
inside that event, not merely that the row belonged to the easy subset.

There is enough positive support to test occurrence/severity supervision.
This does not prove that zeros caused previous failures, that the causal inputs
identify unseen harms, or that a split loss will improve the deployed decision.
The new equal-architecture MSE control is required to distinguish these claims.

[Full counts and source identity](fitting_support.json). Detector-track pixels,
raw-step obs8/pred12, opened development only; not human gold, independent
confirmation, metric, seconds, physical safety, true3D or foundation.
