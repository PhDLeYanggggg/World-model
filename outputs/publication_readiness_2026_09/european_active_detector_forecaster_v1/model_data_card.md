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
conditioning and dimensionless pre-squash output. Its width is64, with4heads
and2layers under the original architecture contract.

There are8ego history tokens and8neighbor histories of8steps. A zero-initialized
8-to64projection adds detector information to ego embeddings only, through a
temporary hook removed on both normal and exceptional exits. The512extra
parameters raise the count from88,514to89,026. Single-process eager Torch is
tested;TorchScript,export equivalence and MPS are not established in this run.

Four fixed arms use disabled,current-box-only,causal identity-key noise and
full detector history. History is the registered primary,not a selected seed.
Current-only repeats current box descriptors and contains no prior box motion.
Equal parameter count does not imply equal information rank.

The preceding72model flat complete-neighbor input experiment remains valid
within its own design. It is not a matched replacement of this actual active
grouped/partial-neighbor producer. There are9actual parent designs,all retained
here;there is no original active complement-parent family to omit.

## Data And Information Boundary

The producer population contains318,969queries over163SOURCE recordings.
Detector input cache IDs,recording IDs,agent IDs and frames align row-for-row.
The verified past-only cache has64fields:eight groups over eight observations,
including log box dimensions,raw confidence,aspect,relative size change,
body-relative center movement,overlap and observation support.

Original agent positions,legal partial neighbor histories,fixed requested
prediction times and causal baseline rollouts are inputs. Eventual future
availability is not the request mask. No future endpoint,future goal,central
velocity,remaining track length,test-endpoint goal or future target latent is
an input. Future coordinates and valid masks are supervised-loss/evaluation
labels only. Normalizers use equal-weighted own-fitting localities,not future
label support or readout statistics.

The readout admits only the old149,510controller-fitting IDs from74recordings,
outside each producer's fitting localities. All12required localities remain.
The89other recordings are not newly scored as controller-held outcomes.
No independent selection,calibration,confirmation or DroneCrowd result opens.
SOURCE has historical development exposure;producer exclusion is not a fresh
independent test. Seeds and overlapping windows are not independent samples.

## Training And Controls

All9parent designs times4arms train4,000steps each:144,000primary updates.
The original batch64,source-balanced draws,loss factors,AdamW schedule,weight
decay and gradient clip remain unchanged. The objective is native available-
label ADE with the original fitting loss factors,not a classification label.
Loss curves are real minibatches,not validation or independent-test losses.

Disabled controls must reproduce full parent scientific states and fresh
predictions. All arms retain the parent's draws,factors and RNG. Pilot work
is900actual updates,with four50+50/direct100recoveries and a parent control.
The first history model receives an additional independent4,000step refit.
These extra engineering checks are separate from primary training counts.

Native arm64 CPU4,interop1,workers0;no resource probing or NumPy training
fallback. Checkpoints every200steps,heartbeats every50steps,explicit resume,
12h boundary,10GiB disk reserve and1GiB private checkpoint cap. No reduction
of rows,models or updates for slowness.

## Evaluation And Safety Limits

The fixed readout preserves every required locality and original label mask,
all6causal physical candidates,hard/easy/complete-future diagnostics,recording
tails,and known/completion-lower/upper paired ADE changes. All six paired
history-versus-physical comparisons are preregistered. Three producer seeds
and3,000paired locality resamples follow fold/seed averaging.

Available-label ADE does not identify the full12step path for partial rows.
Locality082 has no last-step labels in this whitelist;undefined all-locality
FDE/complete-future summaries are not dropped or called zero. A separate raw
audit finds no importer omission or recoverable exact labels on these queries.
Completion bounds are not observed degradations or missing-at-random estimates.

The retained motion bound is not a verified physical safety law. Proposal
accuracy does not establish the approved2%complete all/easy per-recording
positive-harm requirement. Empty selected-risk denominators remain
not_estimable. No learned controller is newly fitted by this experiment.
Any subsequent cost-bank comparison must retain its original producer seed43,
not choose the best seed from the current readout.

## Units,Release And Claims

SOURCE supplies detector-silver bbox midpoints in image-plane pixels. The
protocol is obs8/pred12/stride12 raw frames,not historical SDD raw-frame t+50.
No verified effective seconds,homography,metric scale,human-gold targets or
live sensor-as-of claim. The experiment trains no new scene-image encoder or
JEPA and executes no generative rollout. Not true3D or foundation evidence.

Public artifacts include code,configurations,hashes,losses and aggregate
metrics. Raw tracks,detector caches,prediction coordinates and checkpoints
remain private and excluded from Git. A public anonymous from-scratch asset
package is not completed merely by local exact replay. Private author records
are excluded from anonymous material. No deployment or submission-readiness
promotion,Stage5C execution or SMC follows.
