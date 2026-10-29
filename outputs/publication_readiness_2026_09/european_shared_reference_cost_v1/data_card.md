# Data and Evaluation Scope of the Shared-Reference Ablation

## Published Source and Our Adapter

The source is Wolff and Perry's **Pedestrian Trajectory Dataset of Public
European Squares**, version1.1,16January2026,
[Zenodo DOI10.5281/zenodo.18267205](https://zenodo.org/records/18267205).
The publisher describes39public-square locations and webcam-derived trajectories
using YOLOv10,ByteTrack and georeferencing. Those publisher-level counts and
processed coordinates are not the experimental population or units here.

The pinned acquisition stores publisher code commit
`d7225dd37d6b2bf962ee8533f9ca603593f5c1fc` and raw archive SHA256
`068e584db8895447516d548de7cd3a7b67b1ceaae16afdf76c677602e5e8235f`.
The prior acquisition validated CC-BY4.0 metadata and publisher file checksums.
This round uses the existing verified local asset chain;it downloads no data
and executes no publisher tracking/preprocessing code.

Our input is unsmoothed arithmetic box centers from released raw detector
tracks,not the publisher's cleaned/georeferenced trajectory files. Raw-prefix
tests establish the adapter's annotation-prefix causality,not the entire
publisher detector/tracker provenance. Labels are detector-track/silver,not
human gold. Camera obliquity,perspective,occlusion,tracking errors and missing
future labels remain limitations. This is not SDD official top-down annotation
or a reproduction of an official European Squares forecasting benchmark.

## Task and Source Cohort

The frozen grid observes eight past samples at raw offsets
[-84,-72,-60,-48,-36,-24,-12,0] and requests12future samples at[12,...,144].
Eight-step past support alone determines target eligibility. No future support,
survival,track remaining length or future endpoint defines inference eligibility.
Current-visible short-history agents are retained as masked context by the
source adapter. The current cost model uses its frozen indexed causal features;
this experiment does not expand neighbor coverage or add scene images/goals.

The underlying registered source cache contains318,969past-eligible targets
from41,550sampled recording/frame queries in163training recordings and12
provisional locality groups. It retains193,705complete,118,217partial and7,047
unknown future windows. These are cohort-construction counts,not independent
observations or a full materialization of all raw-data windows.

The present controller study uses the same24verified TRAIN packets and six
producer/controller contexts as the preceding cost experiment. Its fitting/
inner-held readout has74/31unique source-recording identities. The444/186
record/head views repeat contexts and three head seeds;they are not independent
recordings. Do not equate the controller subset to the complete source cache.

## Roles and Exposure

The source's outcome-blind locality-role manifest is frozen under
SHA256`858065b8a14f15a8741cbd9b0f24e5de595a02512b2552709d37b6eeb5e88435`.
This cost ablation reuses source_training only. Source-internal held records
have already been exposed during development and are not a fresh test.
The current study is not whole-pipeline out-of-fold evidence. Independent
model-selection,risk-calibration,confirmation and separate DroneCrowd outcomes
remain unopened;the original access/lineage limitations are not erased.

Existing raw cohort and packet checks are cached_verified. New cost-head
optimization and readout are fresh_run when actually executed. This data card
is a source/context description,not a fresh raw parsing audit or efficacy test.

## Fixed Forecasts, Inputs and Labels

The action bank is constant position,causal finite-difference CV,damping0.90,
damping0.97,history-OLS4 velocity,history-OLS8 velocity,and the frozen original
ADE-only neural forecast at forecaster seed43. Three cost-head seeds do not
mean three newly trained trajectory producers. The existing causal floor
switch remains unchanged;exact floor aliases are not credited as interventions.

Shared R/ER predictions use a token generated from the same past geometry,CV,
floor rollout and past-only floor switch. Physical fitting actions alone set
normalization. Physical-only optimization consumes no neural candidate input
or outcome in its loss. No scene endpoint goal,central velocity,future validity,
ground-truth endpoint or test normalization statistic enters inference.

Known future positions and masks supervise/evaluate native partial-mask ADE.
Unknown outcomes are not zero-error labels. Full12-step and endpoint support
are distinct;the readout retains missing-outcome disagreement completion bounds.
Image-space proximity is not a physical collision or interaction-safety label.

## Claim and Release Boundaries

Distances remain native image pixels,no meter or effective-seconds calibration.
This obs8/pred12/rawstride12 task is not historical raw-frame t+50. No true3D,
foundation,independent-generalization or safe-deployment claim follows from
building the adapter or enforcing shared-reference invariance. Original2%
full-policy/every-record checks remain required;empty risk is not_estimable
and count matching concerns utility only. Stage5C andSMC stay off.

Raw videos/images,third-party archives,feature caches and weights are not
included in Git. Public artifacts contain code,configs,aggregate metrics and
research reports. The public aggregate archive is not a replacement for the
licensed local data and checkpoints required for fresh inference/training.
