# Matched Temporal Auxiliary Cost-Head Training

Registered before real optimizer updates or new development readout. This follows
the positive temporal-information diagnostic, not a deployment-success result.
The old selected cohort did not reliably improve, so that limitation is central.

## Fixed Hypothesis and Comparisons

Can temporally resolved auxiliary supervision improve the original gain/harm
cost head and its controlled intervention utility without violating the original
selected positive easy-harm budget? Train 72 crossed source/head-seed combinations
with three arms each: no auxiliary, row-mean auxiliary, and temporal auxiliary.
Three head seeds are not three new forecaster trainings.

Use the existing width-32 nonlinear MomentHead encoder and bounded five-moment
decoder. Add the same zero-initialized 24-output auxiliary head to every arm.
All primary parameters, auxiliary initialization, source rows, query sampler
draws, AdamW settings and 2,000-update budget match. The no-auxiliary coefficient
is zero; both auxiliary arms use a fixed coefficient of 0.1. No search over
weights, architecture, update budget, validation checkpoints or thresholds.
The unchanged primary objective is the equally weighted moment and three
signed-score quadratic loss with the existing TRAIN scale/RMS preprocessing.

Temporal targets are the 12 observed stepwise neural-minus-reference errors and
reference errors. The row-mean control repeats each TRAIN row's observed mean
over the same observed mask. Auxiliary loss averages channels/observed steps,
then rows within each query, then queries. The mean-positive-step harm is never
substituted for whole-trajectory primary harm. Unknown future rows are excluded
only from supervised draws and remain unknown in subsequent policy evaluation.

## Data and Leakage Boundaries

Reuse the exact 12 exposed source-development localities, original predictor
producer-chain exclusions and whole-recording TRAIN/validation partitions from
the frozen forest experiment. All preprocessing is reconstructed on TRAIN and
matched exactly against the sealed source checkpoint. All TRAIN arrays and
supervision are hashed for resume. Validate that temporal label means reconstruct
the original signed-error/reference targets before any fit.

Inference accepts only a saved model, causal feature matrix and disagreement
envelope. Future targets and validity masks are confined to fitting loss and
offline evaluation. No future endpoints, central velocity, future remaining
track length, test goals or new independent-role readout. Validation outcomes
are not scored by this training entry point. Fixed-final checkpoints are frozen
before a separately implemented, committed readout. No independent test tuning.

## Execution and Recovery

Native arm64 Python, four compute threads, zero DataLoader workers. Use a 100-
update first-source three-arm pilot; compare interrupted/resumed temporal training
against an uninterrupted replay exactly. Estimate full duration and peak memory
before the 216-fit run. Atomic checkpoints every 100 updates retain optimizer,
sampler/Torch RNG state, loss history, input hashes and draw hashes. Resume checks
all identity fields. Preserve completed checkpoints and immutable reports.

Keep the inherited 10 GiB disk reserve and a 256 MiB total checkpoint cap plus
2 MiB atomic-write headroom. Insufficient storage blocks real fitting before
asset loading; do not delete unrelated files or silently weaken the guard.
The inspection phase is cache-free real TRAIN forward/backward engineering only,
with zero optimizer updates and no trained-result claim. Synthetic tests may use
small temporary checkpoints, which are not real-data training outputs.

The local invocation hard limit is 12 hours. Pilot estimates are estimates, not
completion claims. If local scale is infeasible, retain the checkpoints and use
the authorized isolated M3W CREATE workflow after ownership, current jobs,
storage and environment have been checked. Do not use the HPC login node for
science, duplicate an existing job, modify another project or infer job failure
from SSH timeout. No remote submission is implemented by this local entry point.

## Required Readout Before Any Advance

Training completion is not the scientific gate. Freeze checkpoints first, then
read the same recording-held development rows. Report primary signed-score MSE,
original selected-cohort MSE, known and completion-upper easy-risk, full and
same-query matched-count utility, action rates and unknown outcomes. Compare all
three trained arms and retain the original/additive/Poisson/squared-cost strong
controls. A missing strong control is `not_run`, not a passed method comparison.

Use all 12 locality blocks, with three head seeds averaged inside locality, and
3,000 nominal paired bootstrap draws, seed 20261005. Undefined support fails,
not zero-fills. Temporal MSE alone is not an advance condition. Any claimed
improvement requires controlled positive utility, selected-risk preservation at
the unchanged 2% budget, and no loss hidden by a weaker comparator. No additional
future-completeness or label-derived eligibility filter is allowed.

The readout implementation and independent scalar verification must be frozen
before new validation predictions are inspected. If only auxiliary accuracy
improves, retain a negative policy result. Until all required comparisons are
available, no deployment, calibration, confirmation or submission promotion.

## Claim Boundary

These are neural cost-head updates around fixed trajectory predictors, not new
Transformer/JEPA forecaster training or demonstrated neural world dynamics.
Image-local detector-silver, obs8/pred12, rawstride12 only. No metric, seconds,
human gold, physical-safety, true-3D, foundation or CCF-A readiness claim.
Stage5C and SMC remain off. The long-term research goal stays active.
