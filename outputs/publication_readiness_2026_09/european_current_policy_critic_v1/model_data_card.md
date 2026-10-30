# Current-query risk critic: model and data card

## Material Passport

Native Torch risk-estimation experiment over fixed forecasts. Complete36-head
training and72-group readout are fresh_run. Fixed-query heads and source assets
are cached_verified. Separate full replay exactly reproduces all72 groups,
144 fixed-query controls and the canonical summary;see verification.json.
The model is not deployed.

## Model and changed factor

Row-local and selected-set-context critics have30,726 parameters each,462
causal feature fields,original five moments,action indicators,causal envelopes
and count metadata. Six contexts and seeds17/29/43 give36 new heads. Same
initial states,source/query draws,corrected query RMS,point loss,weights and
2,000-step budget as36 verified fixed-query controls. Only selected-query loss
supervision changes to the current decoder's detached choices. Original selected
context remains fixed. No new forecaster,teacher,trajectory residual or rollout.

Positive harm and easy harm use the existing bounded links. Benefit,reference,
easy reference and all seven forecasts stay fixed. The bounds constrain the
head,not real physical safety. Future targets and support are loss/evaluation
fields only. They are not passed to current-action selection or inference.

## Data and roles

149,510 unique target rows,18,766 queries,74 recordings,12 exposed localities.
Repeated context/head-seed views total897,060 rows and112,596 queries;these
are not additional independent observations. EuropeanSquares is an oblique
webcam detector-track SOURCE development dataset,not an SDD benchmark result.
Task:8 observed,12 predicted steps,stride12 raw frames. No seconds conversion.

Labels are detector-silver in local image coordinates. No verified metric scale,
homography or human-gold label. All unknown rows remain in inputs and full
evaluation. No future endpoint,central velocity,future track-remaining length,
oracle best action,test-endpoint goal or test normalization is an inference
field. This contract does not establish whole-pipeline OOF or online perception
causality. Independent calibration/confirmation roles remain closed.

## Measured computation

arm64 .venv-pytorch,CPU4,interop1,workers0,no resource probing/multiprocessing.
Real600-update pilot38.359s,peak9.623GB,exact fixed-control recovery.
Training1877.402s,peak10.798GB,72,000 updates plus4,000 separate refit updates;
both full first-head refits agree scientifically. Full readout628.182s,
peak11.000GB. Separate replay623.356s,peak11.089GB,all controls/groups exact.
Checkpoint100,heartbeat50 plus wall-clock selection heartbeat.
Final step fixed before outcomes;no validation/test threshold or checkpoint search.

## Outcomes and limits

Set-joint known ADE gain6.783336% versus fixed-query8.215347%;paired change
-1.432011pp CI[-3.335655,+0.608111]. Primary completion-lower change
-7.087372pp CI[-13.238339,-2.904036]. Recording risk11/444 supported,313
known easy violations,21 undefined easy views. Four of eight fitting checks
pass;no advancement. [Complete diagnosis](failure_analysis.md).

The3,000-draw bootstrap averages contexts/head seeds within12 exposed localities.
It is conditional exploratory evidence,not exposure-adjusted independent
confirmation. Three critic seeds share one forecast-producer seed. Missing
endpoints remain not estimable;completion bounds are not full labels. No new
physical-bank or SDD-safety comparison is run. Not dynamics/world-model success,
true3D,foundation,metric/seconds,calibrated deployment or submission-ready evidence.

## Availability

Public:code,config,protocol,receipts,aggregate traces/metrics,self-generated SVGs.
Private:not in Git,source arrays,features,history/latent caches,per-row choices,
checkpoints and raw media. Full inference reproduction needs those exact hashed
assets;metric-bundle reduction is a different,aggregate-only check. Stage5C
execution and SMC remain disabled.
