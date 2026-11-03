# EqMotion SOURCE Model And Data Card

## Intended Use

A fixed public-family forecasting control and possible proposal backbone for
future M3W admission experiments. It is not a newly invented EqMotion method,
published best-of-20 reproduction, deployable M3W policy or world-model success.
The original complete recording-risk contract has not been evaluated for a new
policy here. No deployment is upgraded.

## Model

The pinned MIT author core uses hidden width 64, geometric channels 64 and four
layers. Head 0 is fixed before fitting, with one emitted trajectory. Unused heads
remain frozen/stored but are not executed in the formal study. This is direct
deterministic forecasting, not next-token or latent-generative rollout.

Inputs are eight observed ego positions, up to eight complete aligned neighbor
histories, past masks, a causal baseline and the fixed twelve-step request grid.
The third history channel is dimensionless time, not depth. Past-only max-norm
conditioning restores output units before loss/evaluation; its one-unit clamp
limits scale-equivariance claims near zero. There are no images, scene rasters,
goals, future endpoints, central velocity, future masks or test statistics as
model inputs. Scene/goal or JEPA contribution is not proved by this study.

Each model stores 3,027,268 parameters; 580,676 are gradient-enabled, and 386,628
have actual AdamW state through all 4,000 updates. Those counts are distinct.
Inactive paths are not, on their own, an identified performance failure cause.

## Training And Checkpoints

Nine actual fits: three SOURCE folds and seeds 17/29/43. Each fits four SOURCE
localities and predicts the excluded eight. AdamW, batch 64, learning rate
0.0003 and the original cosine schedule, with 4,000 updates per model. Sampling,
label masks, factors and baseline choice match the original PCT. No checkpoint,
seed, head or epoch is selected on excluded outcomes. The first complete fit
and real recovery pilot reproduce scientific state exactly.

The `loss` trace is fitting-site-normalized available-label ADE with the original
indexed/supported correction. The inherited `mean_past_normalized_ADE` trace key
is an unweighted pixel batch statistic on these SOURCE arrays, not a second
normalized validation loss. Minibatch traces are not validation curves.

Local weights remain under
`data/stage_cvpr2027_experiments/european_eqmotion_v1/trials/singleF_seedS/checkpoint.pt`.
Use the exact architecture and baseline index from the corresponding public
fit receipt with `SourceForecaster` and the frozen causal packing functions.
The checkpoint includes optimizer, sampler and RNG state. Do not upload weights
or private caches, feed labels to inference, or substitute arbitrary horizons.

## Data And Evidence

EuropeanSquares released detector/tracker trajectories, not human gold. The
original forecaster SOURCE cohort has 318,969 target windows in 163 recordings
at twelve localities: 193,705 complete futures, 118,217 partial and 7,047 wholly
unknown. These are overlapping agent-query windows, not independent people.
Observation 8, request 12, stride 12 raw frames, image-local pixels. Effective
physical time and metric calibration of these arrays remain unverified.

The current-history-PCT comparison uses the unchanged 149,510-window SOURCE
whitelist, a subset rather than an extra independent cohort. Its partial/grouped
neighbor processing and 64 causal detector-history fields differ from EqMotion.
Original PCT has a bounded output wrapper; EqMotion emits coordinates. Thus
the study is not an isolated equivariance, parameter-count or compute test.

All 72 primary and 144 stronger-control groups reproduce. New EqMotion, original
PCT and current-history PCT forecasts are rerun; dimensionless-PCT arrays are
cached_verified, not falsely called fresh inference. Intervals use 3,000 locality
resamples after producer-fold and actual-seed averaging. SOURCE has already
informed development; this is not independent final confirmation.

## Limits

Observed-label gains are positive, but all completion-lower intervals cross
zero, and locality 082 loses to the current PCT on observed labels. Bounds cover
wholly unknown rows, not missing steps within partial futures. Detection/tracking
noise may be learned along with motion. No complete-policy 2% safety pass,
physical validity certificate, cross-dataset world-model success or submission
readiness follows. Calibration private arrays, final confirmation and DroneCrowd
remain closed. Pixels/raw frames/silver only; not metric, seconds, human gold,
true 3D or foundation. Stage5C and SMC remain off.
