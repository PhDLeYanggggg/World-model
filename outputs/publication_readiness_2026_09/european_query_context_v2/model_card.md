# Query-Context Harm Head: Research Only

## Intended Question

Can contemporaneous query-wide causal context improve the identification of
harmful forecast interventions? This is a cost head, not a new neural forecaster,
latent generator, trajectory correction model or deployable world model.

## Inputs and Architecture

The 380 past-only features include existing motion, neighbor and frozen
forecast-rollout diagnostics. Source preprocessing is fit only on inner-fit
recordings. At each recording/current frame, `peer` receives its own standardized
and clipped vector plus the leave-self-out mean of all available causal rows;
singleton queries receive self. Unknown-future and ineligible peers are retained.
No labels, known-future mask or future-frame rows enter the context operation.

Both arms share the same copied width32 encoder for their two input calls and
use a two-output head over the concatenated representation. The self control
uses its own vector twice with the same parameter count and compute. Each has
12,322 trainable and 25,471 total parameters. A rational bounded decoder updates
only all-harm and easy-harm moments. Benefit and both reference moments remain
bitwise protected from the same frozen shared-utility anchor.

## Learning and Decisions

Six producer/controller contexts, head seeds17/29/43, 2,000 updates per arm.
AdamW, learning rate0.001, weight decay0.0001, clip5, four sources with16 queries
each per update. Original query-balanced moment/signed-score loss, task regret
and scaled dual penalties are unchanged. Whole-recording fit/held partitions
remain within original TRAIN; all twelve localities have prior development
exposure. Checkpoints include model, optimizer, samplers, RNG and data identities.

Inference retains the existing causal eligibility mask and requires positive
predicted benefit minus harm, predicted harm <=2% predicted reference, and
predicted easy harm <=2% predicted easy reference. No threshold was fitted on
held outcomes. Those predicted constraints are not observed safety certificates.

## Outcome and Limits

Peer minus matched self utility: -0.239274pp, nominal95%CI[-0.352816,-0.143938].
Complete support0/72; known easy violations44/72; undefined easy risk12/72.
This model is not deployed. A source preprocessing lookup is not an unseen-domain
adapter. Three head seeds do not meet the three-forecaster main-result requirement.
Mean pooling does not encode explicit agent-to-agent geometry or pairwise losses.

Obs8/pred12 rawstride12, detector-silver, image-local/dataset-local. No metric,
seconds-level, physical-safety, human-gold, true3D or foundation claim. Historical
Stage37 scores remain exploratory and are not recertified. Stage5C/SMC are off.
Weights/data/figures remain private local artifacts; Git contains code, protocol,
hashes and aggregate reports only.
