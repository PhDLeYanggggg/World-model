# Frozen-risk signed-utility decoder: model/data card

## Material Passport

No neural model is newly trained here. Eighteen corrected-unit selected-set
critics and all seven forecasts are cached_verified. Fresh work re-infers fixed
costs and runs four causal decoding policies on all72 SOURCE groups. Separate
full replay exactly reproduces all72 groups,72 controls,148 record/context
oracles and the summary;see verification.json. The decoder is a research
control,not a deployed world dynamics model.

## Model interface

Inputs:finite predicted B,H,R,ER,EH for seven candidates per row,causal candidate
availability,current recording/frame and row IDs. No target,oracle action,
future endpoint,central velocity,future track length or future support enters
the new allocator. Existing462-field feature preprocessing and risk heads stay
frozen. The18 critics cover six contexts and seeds17/29/43;all share forecast
producer seed43. These are not three independently trained forecast pipelines.

Only the B>H per-action legality filter is removed in signed policies. Per-action
positive R/ER,one candidate per row,nonnegative total predicted utility and both
original2% positive-harm/reference inequalities remain. Benefit does not cancel
positive harm in either numerator. Failed numerical solves retain genuinely
feasible incumbents. Optimality is not asserted after a fallback.

The primary comparison fixes intervention count in each current query. The
secondary free-count comparison can make more interventions;it is not an
equal-rate result. Either must still pass complete original recording risk.

## Data and provenance

The same149,510 unique target rows,18,766 queries,74 recordings and12 exposed
European SOURCE localities are reused. Repeated contexts/head seeds are not
independent samples. Labels are oblique-webcam detector-silver in local image
coordinates. Task:obs8/pred12/rawstride12,not historical SDD rawt50 or verified
seconds. No metric scale,homography,human-gold,true3D or foundation claim.

Known-label oracle analysis is label-side only and never exported as an
inference field. All unknown rows remain in full policy evaluation. Available
future-point ADE does not imply complete trajectories;missing endpoints remain
not_estimable. No unknown outcome or risk denominator is imputed safe zero.

## Measured execution

Native arm64 .venv-pytorch,Torch inference CPU4/interop1/workers0,SciPy discrete
optimization. No resource probing,multiprocessing or NumPy training fallback
claim. Real128-query pilot:20.740s,peak9.605GB,1,234 rows,exact interrupted
recovery,without outcome scoring. Full readout:1564.756s,peak11.025GB,all72
original prediction/action/score controls and148 oracle controls exact.
Separate full replay1563.212s,peak11.267GB,all controls/summary exact.
No threshold/model/checkpoint is selected from partial outcomes.

## Outcome and use restriction

Primary matched known difference+0.010488pp CI[+0.002104,+0.019958];primary
completion-lower-0.007179pp CI[-0.024395,+0.007941]. Four of eight fitting
checks pass. Original recording risk1/444 supported,383 known easy violations
and13 undefined easy views. Both label-side oracles are positive feasible on
148/148 record/context views,with148 exactly identical optimal utility values.
The positive-action filter is not a demonstrated label-side bottleneck here.

Intervals are nominal3,000-locality development resamples,not independent or
adaptive-search-adjusted inference. No trajectory-dynamics,new SDD safety,
interaction-path/tail-proxy,public-method superiority or deployment claim.
Independent calibration/confirmation roles remain closed. Stage5C execution
and SMC remain disabled. Submission readiness is false.

Public:code/config/protocol,small receipts,aggregate ZIP,derived tables and
self-generated figures. Private:raw media,source arrays,features,per-row choices,
latents and checkpoints;not in Git. Public aggregate reconstruction is distinct
from full private-asset reproduction. See reproduction_zh.md and failure_analysis.md.
