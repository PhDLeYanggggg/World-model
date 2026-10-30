# Actual Active Producer: Model And Data Card

## Intended Use And Status

This is a controlled SOURCE-development trajectory-input experiment, not a
deployed intervention policy. Numerical training, fixed readout and separate
replay status are recorded in [evidence.json](evidence.json) and
[readout.md](readout.md); the design description alone proves no accuracy gain.
The broad M3W world-model objective remains unfinished.

## Architecture And One Controlled Change

The parent is `DimensionlessAgentTrackSourceForecaster`, the actual producer
consumed by the current cost learner. It retains grouped temporal and
cross-agent computation, partial-neighbor geometry and masks, the original
conditioning and dimensionless pre-squash output. Its width is 64, with 4 heads
and 2 layers under the original architecture contract.

There are 8 ego history tokens and 8 neighbor histories of 8 steps. A zero-
initialized 8-to-64 projection adds detector information to ego embeddings only,
through a temporary hook removed on both normal and exceptional exits. The 512
extra parameters raise the count from 88,514 to 89,026. Single-process eager
Torch is tested; TorchScript, export equivalence and MPS are not established.

Four fixed arms use disabled, current-box-only, causal identity-key noise and
full detector history. History is the registered primary, not a selected seed.
Current-only repeats current box descriptors and contains no prior box motion.
Equal parameter count does not imply equal information rank.

The preceding 72-model flat complete-neighbor input experiment remains valid
within its own design. It is not a matched replacement of this actual active
grouped/partial-neighbor producer. There are 9 actual parent designs, all retained
here; there is no original active complement-parent family to omit.

## Data And Information Boundary

The producer population contains 318,969 queries over 163 SOURCE recordings.
Detector input cache IDs, recording IDs, agent IDs and frames align row-for-row.
The verified past-only cache has 64 fields: eight groups over eight observations,
including log box dimensions, raw confidence, aspect, relative size change,
body-relative center movement, overlap and observation support.

Original agent positions, legal partial neighbor histories, fixed requested
prediction times and causal baseline rollouts are inputs. Eventual future
availability is not the request mask. No future endpoint,future goal,central
velocity,remaining track length,test-endpoint goal or future target latent is
an input. Future coordinates and valid masks are supervised-loss/evaluation
labels only. Normalizers use equal-weighted own-fitting localities, not future
label support or readout statistics.

The readout admits only the old 149,510 controller-fitting IDs from 74 recordings,
outside each producer's fitting localities. All 12 required localities remain.
The 89 other recordings are not newly scored as controller-held outcomes.
No independent selection, calibration, confirmation or DroneCrowd result opens.
SOURCE has historical development exposure; producer exclusion is not a fresh
independent test. Seeds and overlapping windows are not independent samples.

## Training And Controls

All 9 parent designs times 4 arms train 4,000 steps each: 144,000 primary updates.
The original batch size 64, source-balanced draws, loss factors, AdamW schedule, weight
decay and gradient clip remain unchanged. The objective is native available-
label ADE with the original fitting loss factors,not a classification label.
Loss curves are real minibatches,not validation or independent-test losses.

Disabled controls must reproduce full parent scientific states and fresh
predictions. All arms retain the parent's draws, factors and RNG. Pilot work
is 900 actual updates, with four 50+50/direct-100 recoveries and a parent control.
The first history model receives an additional independent 4,000-step refit.
These extra engineering checks are separate from primary training counts.

Native arm64 CPU 4, interop 1, workers 0; no resource probing or NumPy training
fallback. Checkpoints every 200 steps, heartbeats every 50 steps, explicit resume,
12h boundary, 10 GiB disk reserve and 1 GiB private checkpoint cap. No reduction
of rows, models or updates for slowness.

## Evaluation And Safety Limits

The fixed readout preserves every required locality and original label mask,
all 6 causal physical candidates, hard/easy/complete-future diagnostics, recording
tails, and known/completion-lower/upper paired ADE changes. All six paired
history-versus-physical comparisons are preregistered. Three producer seeds
and 3,000 paired locality resamples follow fold/seed averaging.

Available-label ADE does not identify the full 12-step path for partial rows.
Locality 082 has no last-step labels in this whitelist; undefined all-locality
FDE/complete-future summaries are not dropped or called zero. A separate raw
audit finds no importer omission or recoverable exact labels on these queries.
Completion bounds are not observed degradations or missing-at-random estimates.

The retained motion bound is not a verified physical safety law. Proposal
accuracy does not establish the approved 2% complete all/easy per-recording
positive-harm requirement. Empty selected-risk denominators remain
not_estimable. No learned controller is newly fitted by this experiment.
Any subsequent cost-bank comparison must retain its original producer seed 43,
not choose the best seed from the current readout.

## Units,Release And Claims

SOURCE supplies detector-silver bbox midpoints in image-plane pixels. The
protocol is obs8/pred12/stride12 raw frames, not historical SDD raw-frame t+50.
No verified effective seconds, homography, metric scale, human-gold targets or
live sensor-as-of claim. The experiment trains no new scene-image encoder or
JEPA and executes no generative rollout. Not true 3D or foundation evidence.

Public artifacts include code, configurations, hashes, losses and aggregate
metrics. Raw tracks, detector caches, prediction coordinates and checkpoints
remain private and excluded from Git. A public anonymous from-scratch asset
package is not completed merely by local exact replay. Private author records
are excluded from anonymous material. No deployment or submission-readiness
promotion, Stage5C execution or SMC follows.
