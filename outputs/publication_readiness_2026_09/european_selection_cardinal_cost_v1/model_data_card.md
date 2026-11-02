# Model And Data Card

## Model

This experiment trains a continuous five-cost critic,not a new trajectory
generator. It uses the original29,253-parameter SOURCE context CostHead:444causal
fields plus candidate code,a64wide encoder,same-query other-target aggregation,
shared reference/easy-reference and coherent benefit/harm outputs. The seven
candidate paths and all SOURCE forecasters are frozen. Action inference applies
the original predicted positive-gain and2%all/easy harm guards. No class oracle,
future endpoint,future availability or future goal is an inference input.

Three new objective arms by nine actual producer/seed combinations are compared
against nine hash-verified original context critics. Objective changes are not
architecture gains or new JEPA/Transformer dynamics. No new scene image,goal,
camera-ID or scene-ID input is introduced. Original partial observed neighbors
and past detector history remain. Same-query pooling is not explicit pairwise
collision/conflict optimization or physical safety verification.

## Data And Roles

EuropeanSquares-derived detector-silver,local-image coordinates,obs8/pred12,
stride12rawframes. All geometry and detector history are current/past-only.
Original SOURCE controller whitelist:149,510rows from74recordings/12localities.
Per forecaster,the critic fits91,562/145,422/62,036rows according to SOURCEfold,
at eight sites excluded from that forecaster's four training sites. Three seeds
17/29/43 repeat these fitting scopes,not new independent scenes. Complete query
coverage is fixed to at least four visits,with source-balanced query sampling.

The six already exposed model-selection localities contain38,102target rows
across28recordings. No target is dropped for future completeness. There are
37,254known-outcome rows,including15,820partial futures and21,434complete12step
labels;848wholly unknown rows remain. Endpoint labels exist on27,694rows.
Masks enter loss/evaluation only; complete12step filtering is prohibited.

All learned normalization and loss scales use fitting SOURCE only. Selection
labels never fit weights,preprocessing or thresholds. Calibration,confirmation
and DroneCrowd remain closed. SOURCE fitting-resubstitution is not validation;
selection-bootstrap evidence is development,not untouched test confirmation.

## Evaluation

Registered primary:mean_absolute minus frozen weighted_centered independent
policy paired completion-lower utility. Known-label and other loss arms are
reported alongside it. Three real training seeds and3000locality bootstrap
draws;overlapping windows and dependent producer views are not independent.
Original positive all/easy harm/reference and easy-net2%requirements apply to
every required recording. Zero-reference risk is not estimable,not safe zero.
Available-label ADE and diagnostic endpoint FDE do not identify missing future
steps or metric/seconds-level accuracy.

## Runtime And Artifacts

Native arm64Torch,CPU4/inter-op1/workers0. Every100completed updates has an
atomic private checkpoint and scalar heartbeat. SIGTERM/interruption preserves
the last completed checkpoint,not a half-update snapshot. Recovery pilot and
separate complete refits test exactness; numerical replay is distinct from
generalization evidence. Private rows,predictions and weights stay outside Git.
Public artifacts contain code,configs,hash receipts,scalar metrics,derived plots
and documentation only. See phase receipts for actual completion and resources.

## Limitations And Deployment

Neither lower cost loss nor a positive mean gain certifies the selected subset.
Asymmetric loss and centered ranking are hypotheses tested here,not proven
unique causes. Rare harms,reference inflation,partial labels,clipped inputs,
finite scene support and new-scene shift can all remain. No new joint-allocation
result is established. Deployment stays unchanged regardless of an isolated
secondary improvement. Current evidence is2.5D/pseudo-3D trajectory-state work,
not true3D,metric,seconds,human gold or foundation success. No Stage5C/SMC.
