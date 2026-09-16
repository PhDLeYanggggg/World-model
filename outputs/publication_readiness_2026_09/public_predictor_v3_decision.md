# Matched Public-Core Predictor Development Study

Frozen before this study's fitting and development predictions. The user
approved observation 8 / prediction 12, raw-frame t+50 supplemental, and
delegated the research route. Historical exposure is not repaired by this study.

## Question

Does an equivariant public forecasting core supply useful candidates where the
local Transformer has little floor-relative headroom? Compare the pinned MIT
EqMotion core, fixed head 0 (K=1), with the local Transformer. This is not an
EqMotion published best-of-20 reproduction, a generative rollout, or SMC.

## Frozen Comparison

- Same prior fit/development recordings, three physical fit folds, seeds 17/29/43.
- Same native annotation-grid obs8/pred12 labels, past-only coordinate transform,
  fit-only statistics, causal CV reference floor, primary past-normalized ADE,
  equal physical-scene aggregation, easy ceiling 2%, and two fixed policies.
- Both predictors use only complete, timestamp-aligned past neighbors. Missing
  future neighbors and future target masks cannot determine context eligibility.
  Cost heads retain the same original past-context features for both backbones.
- Each full/fold forecaster: 10,000 AdamW updates, batch 32, learning rate 0.0003,
  Smooth-L1 beta 1, gradient clipping 1. This matches update/sample budgets, not
  parameter count or floating-point compute. Tail batches remain shorter.
- Local Transformer width 64, four heads, two layers. EqMotion hidden/channel
  width 64, four layers, author arithmetic unchanged, fixed emitted head 0.
- Same cross-fitted ridge and 1,000-update neural benefit/harm heads. No teacher
  from Stage37 and no historical checkpoint warm start.
- A 100-update fit-only cost pilot continues by identity-checked resume into the
  registered budget; it is not a completed predictor result or model selection.
- CPU arm64, four compute threads, one interop thread, zero workers initially.
  Healthy training may run slowly; device changes must be recorded in resume
  history and cannot silently change architecture, data, loss, or budget.

Compared with v2, budget, batch size and past-neighbor eligibility change together;
v2-to-v3 is not a single-factor causal comparison. The two v3 backbones provide
the matched-budget contrast. Checkpoints, heartbeats and large caches stay local.
Every completed fixed-budget seed is reported, including negative results.

## Claims Not Established

Students01/03 are one physical development scene, not two independent sites.
Three seeds measure training variability; they do not supply a scene confidence
interval. Raw t+50 remains separate and not run here. Native coordinate errors
are reported per recording, not pooled as meters. No seconds, true-3D,
foundation, independent-confirmation or deployable-safety claim is authorized.
Public best-of-20, actual-count-matched intervention, cost-sensitive deferral,
independent calibration/confirmation and scene/goal contribution remain separate
evidence gaps. Stage5C and SMC remain disabled.
