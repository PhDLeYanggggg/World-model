# Detector-History Trajectory Producer: Model And Data Card

## Purpose And Current Status

This is a controlled input study, not a deployment promotion or a complete
multimodal world model. Past detection-box measurements are supplied to the
trajectory producer, following a negative experiment that supplied them only
to intervention-cost heads. Full training is in progress at this card's first
commit; the eventual readout and evidence receipts determine its result.

## Architecture

The inherited causal producer has width 64, four attention heads, two layers,
eight ego-history tokens and 72 aligned neighbor-context tokens. It predicts
12 relative positions, restores the current image origin, and retains the
original motion-bounded output parameterization. A zero-initialized 8-to-64
projection adds 512 parameters to the ego tokens, for 89,026 parameters total.
It does not add a future-input token, RGB encoder, JEPA target, generative
rollout, learned routing head or new correction specialist.

The inherited motion budget is zero for an exactly stationary ego history.
Such a row stays on its stationary physical floor even if the future begins
moving or past box shape changes. These misses remain in loss and readout;
this controlled input study does not quietly change that architecture limit.

The four registered arms are disabled, current-only, identity-key noise and
full eight-step detector history. History contains shape, confidence,
body-relative motion and box overlap together; an improvement would not identify
a confidence-only mechanism. Equal parameter counts do not mean equal input
rank. Original causal trajectories and neighbor support remain present.

## Data And Roles

The complete producer population contains 318,969 rows from 163 European SOURCE
recordings and 12 localities. These are supplied detector-silver bounding-box
midpoints, not human gold. Observation length is eight steps, prediction length
12 steps, and sampling stride 12 raw frames. Coordinates remain image pixels.
Effective seconds, metric geometry and live sensor-as-of causality are not
certified. The supplied-export past-only lookup is checked separately.

Each of 18 original single/complement fold-and-seed jobs uses only its original
fitting localities. Input normalization uses fitting-locality moments, with
equal locality weights, and never filters on query future-label availability.
The sampler, training targets, valid masks and own-fit loss factors are inherited
unchanged. Training labels enter the loss, not the inference feature vector.
There is no central velocity, future endpoint input or test-endpoint goal map.

The readout whitelist is the old 149,510 controller-fitting rows from 74 records.
For each producer, only rows outside its fitting localities can be scored.
The other 89 recordings are not newly scored. Producer supervision within its
original fitting localities is not a readout of those recordings. SOURCE has
historical development exposure; excluding a locality here does not restore
independent model-selection, risk-calibration or confirmation evidence.

## Training And Controls

The fixed experiment comprises 72 real Torch producers, three seeds
(17, 29, 43), all 4,000 updates per model and 288,000 primary updates. Batch size
is 64. AdamW uses learning rate 0.0003 with the inherited cosine schedule and
weight decay 0.0001; gradients are clipped at five. Recorded gradient norms
are measured before clipping. The real pilot adds 900 engineering updates;
one complete history-model refit adds 4,000 reproducibility updates. They are
reported separately, not counted as new primary models or new data.

All 18 disabled endpoints must match their original weights, optimizer,
sample draws, RNG state and loss traces. Every active arm must have genuine
detector-projection updates. The first full history model must refit exactly.
Inference also compares disabled with a freshly loaded original model on
identical IDs and batch partitions. Exact local recovery is an engineering
property, not independent scientific confirmation.

## Evaluation And Limits

Single-fold history is the fixed primary result. Complement models are
secondary controls, not replacement winners. The readout retains all six
physical baselines and each producer's own-fit-selected physical reference.
It reports masked ADE, endpoint FDE, observed hard/easy subsets, complete-future
rows, locality and recording diagnostics, and 3,000 locality-bootstrap draws
after averaging fold/seed views. Overlapping windows are not independent units.
The 72 arm receipts are not 72 independent data populations: paired arms share
18 producer designs, three seeds and the same 12 locality clusters.

Partial labels do not identify full 12-step ADE. Wholly unknown rows retain a
causal forecast-disagreement completion envelope; zero reference mass remains
undefined. Hard/easy membership based on observed errors is diagnostic, not
identification of unobserved labels. Future-informed oracle opportunity is
never an executable model. A proposal-accuracy gate cannot certify the original
2% complete all/easy recording-risk requirement for a learned switching policy.

Raw data, weights and caches stay local. Public code, configuration, hashes,
aggregate receipts and figures are not yet a complete anonymous from-scratch
reproduction package. No true-3D, foundation, metric, seconds-level, Stage5C
execution, SMC or submission-readiness claim is made.
