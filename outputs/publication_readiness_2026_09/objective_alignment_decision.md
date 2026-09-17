# Registered Fit-Only Training-Objective Comparison

## Material Passport

Status: design fixed before new fitting. Scope: code experiment, offline annotated
8-to-12 forecasting on the approved fit cohort. This is not independent test
evidence or a change to the primary evaluation. User-delegated research execution.

The preceding 36 fits lower training log-loss but fail the primary held-scene
ADE. Uniform-window log1p(ADE) training differs from equal-scene mean ADE
evaluation. This study isolates sampling and loss, then adds a fixed unit-weight
positive-harm penalty relative to CV. No claim that alignment must succeed.

Five arms: row_log, row_ade, scene_log, scene_ade, scene_ade_harm. Same frozen
geometry-network architecture, same inputs including neighbor history and baseline
rollouts, same random initialization, seeds17/29/43, three physical fit-scene
folds, all11,966rows. No added RGB, no new goal data or future input. Geometry
forward is checked exactly against the old network. Constant training dimensions
are zeroed after train-only standardization in every arm; this common change is
not attributed to the loss. Scene sampling assigns equal total probability to
each of the two training sites, with uniform draws inside each site.

Each arm receives4,000AdamW updates, batch64, learning rate0.0003, weight decay
0.0001, gradient norm cap5, final-update checkpoint only. Pilot stops at100
updates and resumes the same fit; it never evaluates held data. No interim held
checkpoint choice, alpha search or fallback threshold fitting. No early stopping
for a negative score. Forty-five fits (180,000updates); no claim of large scale.

The 2x2 core separates log versus mean ADE and row versus scene sampling.
The fifth arm adds mean(max(ADE_model-ADE_CV,0)) to ADE, coefficient1 fixed here.
Future labels are used only in train loss or held-fit evaluation. No hard/easy
ground-truth flag enters inference. Training losses differ and must not be
compared as numerically interchangeable objectives.

Primary remains past-normalized ADE, equal physical-scene aggregation. Report
native-coordinate diagnostic separately. Existing easy threshold unchanged.
Report gain versus CV and train-selected strongest, easy degradation and absolute
harm, tails, stationary-history slices, three-seed variation, paired2,000scene
resamples and binary CV/candidate oracle headroom. The oracle is diagnostic,
not usable at inference. Same scene means are used for every comparison.

All source and cache hashes must verify. All Zara recordings remain one physical
site. Offline interpolation and historically exposed fit roles remain disclosed;
these three sites cannot certify independent risk coverage. Students development,
calibration and confirmation are not admitted to this study. No deployment is
selected from held-fit results. Stage5C and SMC stay disabled.

Falsification: if neither mean-ADE training nor scene weighting produces useful
held prediction/headroom, objective mismatch alone cannot justify more threshold
search. If in-sample gains rise but held gains fall, investigate data/context
support and representation before promoting the model. A successful fit-only
contrast would still need a separate frozen downstream/independent evaluation.

All computation local-first, native arm64 PyTorch CPU4/interop1/workers0, with
optimizer/RNG checkpoint and heartbeat. Use CREATE only if measured costs require
it and access is available; do not pretend an unavailable HPC job was submitted.
