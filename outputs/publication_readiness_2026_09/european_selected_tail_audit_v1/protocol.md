# Frozen Selected-Tail Error Accounting

## Material Passport

Mode: run, source TRAIN-internal posthoc mechanism diagnosis. Prior exposed
results motivated this design; committing it before new computation does not
make it prospective independent evidence. No new training, model selection,
threshold search or deployment. The long-term research goal remains active.

## Question and Fixed Design

Reuse all six controls and144 views of the completed isolated-harm experiment.
Freeze predictions, row order, source/recording roles, eligible pool,2% risk
budget and all model weights. Replay prediction hashes and selected counts.
Read only original TRAIN packets, with fit/held recording partitions unchanged.
No outer development or independent selection/calibration/confirmation access.

For each all/easy risk constraint on each fixed known-label cohort:

observed excess = predicted excess + harm underestimation +0.02 reference overestimation.

Compute exact additive sums after dividing by the frozen source TRAIN scale.
Compare eligible, own-selected and anchor-selected cohorts. The anchor cohort is
identical across arms. Unknown outcomes are counted, never used to filter
inference; existing completion-bound results remain authoritative for policy
safety. Known-only component replacement is a future-informed explanation,
not an inference rule, new selector or deployment result.

## Three Distinguishing Diagnostics

1. Replace only the harm or reference component by its observed value, keeping
   the original selected cohort. Count unsafe known rows still passing each
   risk check. This separates arithmetic error contributions, not causal effects
   or the generalizability of a learned repair. No new policy is promoted.
2. Standardized causal feature radius uses frozen inner-fit mean/std, without
   clipping. Its50/90/99 percentile boundaries use only eligible inner-fit rows.
   Apply those boundaries unchanged to held rows. Radius is a coarse support
   proxy, not nearest-neighbor coverage, density, or an OOD guarantee. Duplicate
   quantiles create empty bins; retain them. Outcomes cannot define these bins.
3. Compute the exact supremum of isolated all/easy harm under its bounded
   +5/+5 log-odds shifts, given the frozen anchor and causal envelope. Compare
   observed labels against that supremum. A single realized value above a
   conditional-mean model's ceiling is NOT proof of irreducible predictive bias;
   report cohort mass and shortfall as architectural feasibility diagnostics.

Preserve all views and sources, no favorable subset choice. Counts repeat rows
across contexts/seeds. No rowwise uncertainty or independent sample claim.
All aggregation is descriptive; no newly invented success gate or CI is needed
for the exact accounting identity. Prior locality-bootstrap intervals stay
nominal and exposed TRAIN-internal.

## Execution

Native arm64 local CPU4, workers0. Reuse hash-verified local checkpoints and
stream approved CREATE TRAIN packets read-only; do not duplicate scheduler
jobs. Atomic per-view receipts, heartbeat, explicit resume,12h limit. Never
overwrite complete outputs. Preserve prior source code and checkpoint hashes.
Public output contains aggregates only, not histories, labels or row-level
predictions. Private runtime logs are ignored by Git.

Observation8/prediction12 at raw stride12, detector tracks and local/pixel
coordinates. No metric, seconds, physical-safety, human-gold, true3D or foundation
claim. Stage5C and SMC stay disabled. Historical Stage37 is exploratory, not a
recertified safety guarantee. Failure remains a failure.
