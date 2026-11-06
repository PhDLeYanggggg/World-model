# Prefix-Balanced Actor Component

## Status

Completed research component,not deployed. The whole-forecast progression
screen fails. Original risk rules and previous deployment stay unchanged.
No independent test,foundation model,true3D or physical-safety claim.

## Model And Training

Six source-excluded deterministic EqMotion fixed-head K1 forecasters;
3,027,268parameters each,four layers,hidden/channels64,one seed17.
Original author core and architecture are pinned by registration_v2.json.
No third-party code is newly committed. Each model receives4000AdamW updates,
batch64,learning rate0.0003with original cosine decay and gradient clip5.

Input is eight past image-coordinate positions and up to eight complete,
aligned neighboring histories. The prediction interface requests twelve
future annotation steps. Future targets/masks enter loss/evaluation only.
It is an ego-agent predictor with neighbor context,not a jointly rolled-out
world simulator. No JEPA contribution or new Transformer is claimed here.

The sole change is fixed prefix-balanced supervision. Same training-only
normalizers,selected motion baseline,initial weights,sampling and budget as
old-loss controls. Checkpoints contain optimizer and RNG states,not just weights.
One full old-control refit and a real100+100vs200recovery test reproduce exactly.

## Data And Evidence

125540existing producer-A windows,14source groups,34recordings from admitted
MovingDroneCrowd++ fitting roles. Every teacher excludes its own donor sources.
Roles rotate across contexts; these are not newly independent test scenes.
122716rows have labels,90965complete12,31751partial and2824wholly unknown.
No future endpoint,central velocity,test endpoint goal or test-stat normalization
enters inference. No new own-B,C/H/reserved images or labels are read.

Equal-source paired known gain differences vs old forecaster are+12.439pp at
prefix1but only+0.098pp at available12;6/14sources improve at12. Conservative
paired lower is-9.474pp,exploratory interval[-17.809,-3.083]. This bound is not
the true unknown effect. All34new raw-actor known-easy harm ratios exceed2%.
No favorable prefix or endpoint statistic replaces the failed primary result.

All six prediction banks and analysis scalars reproduce exactly in a zero-update
replay. This is computational verification,not three-seed or independent
scientific evidence. There is no complete learned-policy safety certificate.

## Use And Limitations

Checkpoints remain in the local private experiment directory; raw data,caches,
weights,images and third-party sources are not Git artifacts. Reproduction
commands and recovery checks are in operation_zh.md. Deployment of this new
actor is not approved. A later cost head would require newly recomputed causal
rollout labels,not old labels reused under the new model name.

Image coordinates and raw annotation frames are not verified meters or seconds.
Labels/proxies are not relabeled human gold. Missing-label uncertainty,source
shift,model capacity/target tradeoffs and causal observation insufficiency
remain open. Prefix weighting is a conventional repair,not a new contribution
by itself. Stage5C execution and SMC remain disabled. Submission is not ready.
