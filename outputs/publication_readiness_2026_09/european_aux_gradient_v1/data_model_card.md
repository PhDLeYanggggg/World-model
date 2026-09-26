# Data and Diagnostic Model Card

## Purpose

This study tests a local optimization mechanism behind the failed auxiliary
cost experiment. It does not train or release a replacement trajectory model.
There are no newly evaluated deployment decisions or new forecasting scores.

## Inputs and Supervision

The inherited estimator receives 383 native causal features and a causal
disagreement envelope. The GELU64 shared encoder feeds four nonnegative cost
moments and a separate binary risk head. Targets are nested, row-locality-
excluded forecasting costs. Future trajectories and easy/cap labels enter
losses and diagnostics only. They are never concatenated to inference inputs.

Fitting preprocessing, RMS loss scales and known-cost support are inherited
bit-for-bit. Zero-envelope rows remain eligible for the main cost update;
auxiliary labels are masked where undefined. True and shuffled labels have
the same within-locality prevalence and missingness. No test endpoint goals
or centrally differenced official velocity is introduced by this diagnostic.

## Experiment Units

Three seeds, two feature families, six overlapping producer/controller
assignments, four excluded-locality contexts per assignment yield 144 views.
Three previously trained arms give 432 frozen step-2000 states. Each virtual
branch starts from the identical model and AdamW state for its comparison.
No update overwrites a frozen model or accumulates into a deployable model.

The five interventions each receive eight 256-row update batches per state.
Probe sets contain 256 other known rows per fitting locality. Disjoint rows
are not disjoint trajectories: windows may overlap, and all these fitting
populations have already been exposed to the trained models. Repeats,
overlapping models and windows are not additional independent test units.

## Claims and Limitations

The legacy per-view field outer_outcomes_read=false means no outer-outcome
diagnostic evaluation or use in updates. It is not a filesystem-level blinding
claim: the shared source loader exposes source containers, and fitting_inputs
slices fitting IDs before indexing target_eval and baseline-error labels.
Source localities also rotate fitting/excluded roles across contexts. This
does not open the separate independent selection/calibration/confirmation
assets and does not make the already exposed source data independent.

The reported uncertainty is descriptive locality variation of fitting-probe
effects. It is not evidence of external transfer, risk calibration, general
causal attribution, physical safety or deployment readiness. Final-state
diagnostics do not describe intermediate training. A good binary-risk head
does not establish a good cost-magnitude estimator. Four-cost gradient
projection does not individually protect easy-harm under AdamW.

Protocol: obs8/pred12 native annotation steps in detector-image coordinates.
No metric distance, elapsed-seconds, human-gold, true3D or foundation claims.
Independent selection, reserved calibration and confirmation remain closed.
Stage5C and SMC are disabled. Existing deployment status is unchanged.

## Reproduction and Storage

See operations.md, the immutable original registration and the explicit
display-only implementation amendment. Private data/checkpoints/row-level
diagnostics remain uncommitted. Only code, configuration, reports, hashes and
aggregate statistics are public. Verification distinguishes synthetic tests,
real numerical intervention replay, parent cached verification and scientific
evidence. A failed scientific screen remains failed even when replay passes.
