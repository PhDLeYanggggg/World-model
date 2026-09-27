# Matched Refit Data and Model Card

## Population and Roles

318,969 complete-history target queries, 163 recordings and 12 admitted
European source localities. These are detector-derived image-local pedestrian
tracks, not human-gold annotations. The source task is eight observations and
twelve requested future steps, sampled at raw-frame stride12. No calibrated
seconds, homography, meter-per-pixel or metric 3D claim follows.

Three fixed producer folds each fit four localities and forecast the remaining
eight. Training statistics, baseline choice, loss weighting and easy/hard
cuts use only that producer's four fitting localities. Each locality has two
held producer contexts and three seeds. This is development cross-fitting on
already exposed source data, not restored independent test evidence. Reserved
selection, calibration and confirmation are not opened.

## Causal Inputs

Complete ego position/time history, up to eight nearest current-visible
neighbors, per-step neighbor validity, the declared causal baseline and
requested future query times. Missing observations are masked, not inferred
from labels. All neighbors retain only their current or past recorded slots.
The input schema rejects future endpoints and target-validity masks. Future
trajectory labels enter the training loss and evaluation only.

The intervention changes neighbor eligibility, nearest-neighbor membership,
attention support and observed-token conditioning together. A newly eligible
nearby short-history agent can replace a farther complete-history agent.
This is not guaranteed to increase the count of valid history slots. The
experiment tests the registered mechanism as a whole; it cannot separately
identify information, neighbor membership and normalization effects.

## Model and Training

Same width64, four-head, two-layer past-context Transformer as the matched
legacy control, with deterministic twelve-step forecast queries. There are
no scene images, new goal labels, JEPA pretraining, latent generative rollout,
next-token model or SMC in this experiment. Neighbor tokens are flattened;
the model has ego/neighbor modality embeddings but no explicit persistent
neighbor-identity embedding. That architectural limitation is not resolved
by making incomplete histories visible.

The output starts exactly at the training-selected causal baseline. A radial
squash bounds each correction by the observed ego path or baseline extent
scaled by the requested-step fraction. A stationary observed history cannot
leave a stationary floor under this bound; subsequent movement stays in the
evaluation population. Bounded output is not calibrated physical safety.

Seeds17/29/43, nine new models, 4,000 updates each, batch64, unchanged AdamW
schedule and masked trajectory loss. Equal-locality sampling and factors are
matched to cached controls. A fresh legacy control verifies the full endpoint
and pilot-resume path exactly. No best seed, epoch or threshold is selected.
Reported means average separate producer evaluations; they are not the
performance of an ensemble selected for deployment.

## Reporting and Limits

Source predictions freeze in Git before comparative readout. Equal-locality
percentage gains are computed before averaging, avoiding a pooled-pixel score.
ADE uses available requested labels; FDE requires the final requested label.
Missing future labels are not treated as perfect predictions. Zero-reference
percentages remain undefined; absolute costs remain visible. Easy and hard
are retrospective evaluation subsets, never inference features.

Three thousand locality bootstrap draws describe conditional source
uncertainty. Overlapping windows, shared producer fits and repeated development
exposure prevent interpreting them as an independent deployment guarantee.
Data duplication beyond the admitted source lineage, detector identity errors,
localization noise, incomplete neighbors and uncalibrated scale remain limits.
The new model is an experimental trajectory forecaster, not a complete
multimodal world model. Deployment is unchanged; Stage5C and SMC remain off.
