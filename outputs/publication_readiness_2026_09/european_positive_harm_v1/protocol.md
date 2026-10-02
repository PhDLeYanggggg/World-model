# Positive Conditional Harm Learning

Prospective source-development experiment, not independent confirmation.
The preceding frozen-component diagnosis found harm-driven unsafe expansion.
This experiment tests a combined change in output form and loss. It does not
isolate positivity alone, and nonnegative predictions are not safety bounds.

## Fixed Controls

- Original forest, unchanged.
- Two-harm additive control, using hash-verified H/EH coefficients of the
  previously trained seven-feature auxiliary. Those target-wise ridge fits
  are separable; B/R/ER corrections are zero. No new fitting in this arm.
- Positive total/easy harm: original leaf H/EH times a conditional exponential
  tilt normalized to have training-weighted mean one in that leaf. Fit two
  seven-feature slopes per frozen leaf. Freeze original raw B/R/ER and routing.

All arms use the same original feasibility projection. That projection couples
harm and benefit; projected benefit can change and must be counted. An all-zero
training-harm leaf stays unchanged, not filled with invented labels.

## Training

Use the exact 72 frozen source heads, seeds17/29/43, whole-recording 70/30
source split and known-training query weights. Train-only quality normalization
and clipping[-8,8] stay identical. Seven features: motion-line residual,
last-FD/OLS8 disagreement, width variation, reversal fraction, prefix presence,
prefix detector confidence and partial-neighbor count. No future quality,
sample filtering, teacher refitting, new neural dynamics or tree splits.

Minimize normalized conditional Poisson deviance plus L2/2 with penalty1.
Weighted mean-preserving exponential normalization uses training rows only.
Newton tolerance1e-7, maximum32 iterations, fixed backtracking. Nonconvergence
is an error. Numerical exponent guard[-60,60] is counted, never tuned.
No hyperparameter grid, source/seed winner, threshold search or test access.

## Readout And Advance Screen

Use unchanged moving/support masks, predicted positive utility, and both
predicted harm/reference constraints at 2%. Unknown outcomes remain unknown;
completion bounds are used and undefined/empty support fails. Whole-easy net
degradation is distinct from selected positive harm divided by selected error.

Report signed-score MSE and conservative utility against original and additive,
both full and same-query count-matched. Utility percentages use full known
reference mass, not ADE/FDE improvement. Average fixed views within locality,
then paired locality bootstrap3000, seed20261002. All intervals are nominal
exposed-development intervals. No multiplicity-corrected or test claim.

Advance only if MSE upper CI<0 and full/matched utility lower CI>0 against
both controls, and complete support, number of upper/known risk violations,
and worst upper risk do not deteriorate versus original. Passing this screen
does not certify deployment, physical safety or independent generalization.

## Execution

Real pilot with full fitting/refitting of the first fixed head. Continue all72
without a performance-based stop, provided resources and implementation pass.
Exact fit/serialization/inference/readout replay; checkpoints per completed
head, heartbeat and explicit resume. Healthy work can run up to12hours.
Native arm64, four compute threads, no loader workers or resource probing.
Local numeric cache forbidden while below the unchanged10GiB reserve. Stream
new immutable weights to owned CREATE storage, cap512MiB; no remote training
on login node, no unrelated project interaction, quota remains unknown.

The 12 localities are already exposed development data, obs8/pred12 stride12,
image-local detector-silver. Independent selection/calibration/confirmation
stay closed. No metric/seconds/human-gold/true3D/foundation/publication-ready
claim. Stage5C and SMC remain off.
