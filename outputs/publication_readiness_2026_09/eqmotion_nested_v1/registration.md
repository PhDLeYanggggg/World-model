# EqMotion Pair-Excluded Cost-Training Producers

Registered before new fitting or cache construction, 2026-09-22.
This is a prerequisite experiment, not a new risk-head or efficacy claim.

## Reason and Scope

Frozen Transformer cost heads failed their transferred EqMotion protection
criterion. A new cost head needs forecasts produced without seeing the row's
site or the outer held site. Reusing a model trained on either would invalidate
its training labels. Reuse the established native source protocol; do not change
data roles, principal outcome, risk tolerance or any frozen evaluation result.
The four SDD sites remain research-design exposed, not independent confirmation.

## Fixed Matrix

- Sites: coupa, deathCircle, gates, hyang. Six unordered exclusion pairs.
- Seeds:17,29,43 for each pair. Eighteen new random-initialized EqMotion K=1 fits.
- Architecture and training settings exactly as `m3w_native_eqmotion_v1.json`:
  fixed head0, four layers,64hidden/coordinate channels,4,000updates,batch64.
- Source-uniform samplers, identical train IDs, draw counts and train-only loss
  factors to the completed Transformer pair-excluded producers. Check exact
  sampler equality before accepting each final checkpoint.
- Total budget:72,000updates and4,608,000draws. A100-update pilot can measure
  runtime and resume the same fit; it never substitutes for the complete matrix.
- CPU4/inter-op1/workers0/nativearm64, atomic checkpoint every200updates,
  heartbeat every50updates, explicit resume and single-runner file lock.
- Cache all pair-held predictions, totaling1,581,804row/seed/pair instances;
  175,756unique eligible queries, not1.58million independent examples.
- Reuse all12 verified outer-held EqMotion checkpoints for each head view's
  held-source predictions. No selection by their observed performance.

## Output and Verification

Prediction archives contain only global query IDs and predicted coordinates.
Future labels and gain/harm targets live in separate supervision archives.
Every cost-training view contains the other three sites, each produced by a
model excluding both its own site and that view's held site. Fit-only
normalizers follow the same exclusions. No targets, future validity or
future-derived subset membership enter prediction inputs.

Require all18 fits,12 views,36 ordered training exclusions, exact sampling
matches, known input/code/checkpoint hashes, finite outputs, separate arithmetic
checks of every cached cost and fixed checkpoint replay blocks. Replay checks
the first, middle and last128-row batch per producer, not every prediction.
Unlike the local Transformer wrapper, EqMotion is not forced to emit exact CV
on static history. Do not install that behavior silently during cache creation.

Private checkpoints and arrays stay outside Git. Public code, registration,
lightweight receipts and reports are committed. Interrupted fits resume from
the last atomic checkpoint with restored optimizer and RNG. Interrupted cache
creation reuses verified completed producers; no partial artifact is accepted
as complete. No forecast-gain gate is applied to choose these producers.

## Boundaries

No cost-head fitting, threshold search, original val/test/main/external readout,
independent calibration or deployment occurs in this prerequisite. A later
registered cost-head comparison is still necessary. The task is8observed and
12predicted sampled annotation steps, SDDstride12, annotation pixels. It is not
verified seconds, metric, true3D or a foundation model. Stage5C and SMC stay off.
