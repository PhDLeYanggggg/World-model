# Registered Fit-Only Past-Appearance Forecast Probe

## Material Passport

Question: does point-centered observed appearance improve stationary-start
forecasting over the same causal geometry at an untrained fit site? This is an
exploratory modality probe under the user's delegated multimodal research goal,
not a changed primary protocol, independent confirmation or a new deployment.

The preceding moving-control audit supports local direct-frame-index plausibility,
not exact physical clock or pose labels. We explicitly use that native-index
convention as an input assumption. The ETH CVL source page's research-use scope
was checked in the preceding audit; no raw images, weights or third-party data
are redistributed. The image path is admitted only for this local fit-only probe,
not retrospectively promoted to an official validated sensing modality.

## Fixed Data and Comparisons

Retain all 365 frozen stationary fit windows and labels, 31 agents and 45 runs
at ETH/Hotel. The parent eight observed / twelve predicted steps, split, metric
and easy definition remain unchanged. Fold0 holds ETH, fold1 holds Hotel; Zara
has no stationary rows and remains not_run. This subset is not the full benchmark.
Existing row selection and repeated-window dependence remain disclosed.

Extract centered 96px patches from all eight declared past/current frames,
resize to32x32, fixed RGB/255-0.5 preprocessing. An incomplete patch becomes zero
with an explicit valid mask; no window is removed. Use the frozen28 causal
geometry/neighbor features plus4 past-only supplied-H/local-scene Jacobian
components. Only training rows set feature means/stds; standard scores clipped
at10. No future metadata is permitted in the input cache. Future positions and
start labels are loaded separately for training loss and held-fold evaluation.

Compare geometry-only, geometry+current-RGB and geometry+eight-past-RGB. A shared
small CNN maps each32px image to16 dimensions; the same fusion/head shape sees
eight ordered embeddings, geometry and validity masks. Geometry receives zero
visual embeddings; current-RGB repeats only the last observed embedding. This
matches update and row budgets, not effective parameter count or compute.
No pretrained image weights, pose labels, temporal offset fit or ZNCC features.

Predict12 image-plane displacements with an exactly-zero initialized trajectory
head; map through supplied H to relative native displacements. This explicit
output mapping avoids learning arbitrary camera-to-coordinate axes solely from
one source. It is not physical calibration. Both geometry and visual arms use
the same mapping; only image information changes. The classifier predicts future
annotation-coordinate change, not verified physical intent.

## Training and Reporting

Three seeds17/29/43, two supported folds, three arms =18 models. Each gets1,000
AdamW updates, uniform row minibatches32, lr0.0003, weight_decay0.0001,
gradient_clip1.0. Trajectory loss is smoothed native Euclidean displacement
error / frozen parent scale, multiplied by0.001 for numeric conditioning;
distance epsilon0.001, plus0.1 BCE for the start head. These are optimization
choices, not changes to the reported exact ADE/FDE. No validation/held outcome
selects an update, seed, feature arm or threshold. Report every final checkpoint.

Save optimizer/RNG/loss checkpoints every100 updates and heartbeat/PID. A100-update
partial first-trial run estimates runtime and resumes to the same final budget;
it does not trigger early success or evaluate the held fold. CPU4threads,
interop1, no workers or resource probing, arm64 interpreter. Synthetic exact-resume
and modality isolation checks precede real fitting.

Report unrestricted predictions and the same fixed0.9 probability gate used by
the prior diagnostic; guarded forecasts also require all8 patches supported in
every arm. This gate is not calibrated confidence or a physical safety guarantee.
Missing-context fallback is exact zero/CV. Report parent-normalized and native
ADE/FDE, direction error, start Brier/AUC, switch rate and still/easy absolute
harm. On zero-CV-error easy rows, percentage degradation is undefined, never zero
by convention. Include agent/run-balanced diagnostics; no misleading independent
window CI from two adaptively inspected fit sites.

No final-test use, prospective primary-metric change, historical-score relabeling,
Stage5C, SMC or submission-readiness claim. A positive local probe would motivate
larger independently designed validation, not establish the full research goal.
