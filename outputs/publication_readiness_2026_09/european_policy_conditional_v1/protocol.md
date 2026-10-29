# Frozen-Policy Conditional Excess-Risk Calibration

## Question and Scope

The rational repair eliminated most exact-zero local Jacobians without fixing
selected harm. Does source-excluded conditional calibration improve this frozen
head over global correction and a support-only veto? This is an exploratory
mechanism experiment inside the already exposed original TRAIN data, not a new
official calibration split or independent test. No new trajectory predictor,
neural update, threshold sweep or deployment is allowed.

Reuse all18 rational finals, six producer/controller contexts and head seeds
17/29/43. Each context has four target-source localities excluded from both the
upstream forecaster and old controller fits. The head used only inner-fit
recordings. For each of the four sources, calibrate with the other three
sources' inner-held recordings, then read out the excluded source's inner-held
recordings. No target-source rows supply that calibrator's coefficients.
All72 crossings remain, even when support or utility fails. Different crossings
reuse labels and models, so they are dependent development views. Prior held
exposure is recorded, never erased. Outer development, independent selection,
calibration and confirmation remain unopened by this experiment.

## Fixed Estimator

Freeze raw benefit, harm, all-reference and easy-reference scores, preprocessing,
causal eligibility, producer, controller and the2% budget. Normalize residuals
by the past-only prediction-disagreement envelope, never by future length/error.
Cells are the cross of:

- signed predicted utility/envelope with edges0,.02,.1;
- original all-harm test passing/failing at2%;
- original easy-harm test passing/failing at2%.

These16 fixed causal cells require no fitted quantile or new feature selection.
Within each calibration recording/cell, compute the sum of actual minus
predicted risk-excess (H-.02R and He-.02Re), divided by known-row envelope mass.
Average recordings equally within locality, then the three localities equally.
Keep only the positive part as an additive risk-score penalty multiplied by
each target row's causal envelope. Unknown targets do not become zeros. Record
them, retain them in inference, and retain worst-case completion bounds at
readout. Zero-envelope rows do not provide residual coefficients.

A cell is usable only with at least3 informative recordings,2 localities and
32 known eligible occurrences. These are fixed engineering support minima,
NOT proof of statistical support or exchangeability. Unsupported cells veto
intervention. The global estimator uses the same minima over the whole causal
eligible pool. There is no iterative re-selection or90th-percentile adjustment.
The raw utility ranking stays frozen. Corrections only veto raw actions; they
do not extend selection. Risk penalties are a decision overlay, not a new
coherent five-moment predictor or a conformal/population-safety guarantee.

## Four Controls

1. Raw rational policy, exactly replayed against the archived readout.
2. Global excess-risk correction.
3. Support-only cell veto with zero corrections.
4. Conditional cellwise excess-risk correction (target).

The support-only comparison distinguishes estimation from simply rejecting
unsupported cells. Compare target against all three controls at full coverage
and matched counts inside recording/frame queries, with the same raw utility
ranking and deterministic row tie-break. Do not choose a winning seed or cell.

Freeze every calibrator and all72 decision arrays before outcome reductions.
Targets for excluded-source readout never enter its calibrator or inference API.
Calibrator labels are consumed in other cross-fitted views; this is not a claim
that all held labels are unopened until readout. Check exact raw predictions,
raw policies, per-array hashes, upstream identities and recording exclusions.

## Readout and Advance Rule

Report selected counts, unknown selections, all/easy positive harm, full-policy
completion support, empty/undefined risks, per-recording outcomes, and paired
completion-lower utility/reference mass. This utility is not ADE/FDE. Keep the
existing2% positive-harm and easy-preservation criteria. Empty or undefined
selected-reference policies do not pass. Advancement requires full support on
all72 views and positive lower nominal95% bounds for both full and matched
utility versus every control. No automatic deployment even on a pass.

Use3000 locality bootstrap draws, seed20261029, averaging repeated contexts and
head seeds inside each of12 localities. Cross-calibration induces further
dependence and coefficients are not refitted in this bootstrap. Intervals are
conditional exploratory readouts, not independent or multiple-testing-adjusted
significance. Three head seeds are not three new forecaster seeds.

## Execution and Evidence

Native arm64 CPU4/inter-op1/workers0. Source TRAIN packets remain on CREATE;
stream approved byte/hash-verified packets to local RAM. Save compact frozen
decision/label bundles only under ignored private storage, never in Git.
Preserve10GiB free space; atomic per-view outputs and per-context resume,
heartbeat, exclusive lock,12h boundary. Register code/tests/protocol before data
execution; no new HPC scheduler job. Scheduler timeout is an observation blocker,
not evidence that a job or SSH connection failed.

Results mark cached_verified checkpoints/packets, fresh_run calibration/readout,
and not_run independent transfer. Obs8/pred12, raw stride12, detector-silver,
pixel/dataset-local only. No metric, seconds, human gold, physical safety,
true3D, foundation or submission-ready claim. Stage5C and SMC stay off.
