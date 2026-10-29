# Model Card: Protected Harm Continuation

- Purpose: exploratory prediction of baseline-relative benefit/harm and reference
  moments, not trajectory generation or a deployable world model.
- Data: original European source TRAIN;380-dimensional causal inputs;74 fit and31
  held source/recording pairs,12 localities. Held recordings were previously exposed.
- Design: six contexts, four sources/context, seeds17/29/43;54 new continuations.
- Anchor:18 frozen shared-utility cost heads from the preceding factorial.
- Budget:2,000 extra updates/fit,4,000 effective including anchor;108,000 fresh.
- Arms: utility continuation, joint risk, isolated risk. Same total25,407 parameters;
  active24,615 versus12,258 in the isolated arm.
- Numerics: float32 network/output, float64 bounded odds intermediates; native
  arm64 CPU4/interop1/workers0. No MPS execution claim.
- Optimizer: fresh AdamW, lr.001, decay.0001, clip5; same query sampling continuation.
- Inference: observed-motion/support eligibility, positive predicted benefit-harm,
  and unchanged2% all/easy risk conditions. Unknown source IDs are rejected.
- Supervision: future costs for loss/evaluation only; no future labels or outcome
  availability as inference filters. No central velocity or test endpoint goals.
- Preservation: isolated B/R/Re exactly match the anchor; net gain B-H may change.
- Outcome: isolated full and matched held utility lose to joint risk;47/72 known
  easy-risk violations;3/72 complete policy support. Not deployable or promoted.
- Uncertainty:3,000 locality bootstrap draws, nominal exposed-TRAIN intervals.
- Artifacts: registration, hashed fit/readout receipts, atomic private checkpoints,
  completed logs,83 scoped tests and reproduction commands in conclusions.md.
- Exclusions: raw/cache/checkpoints/figures are not committed. The failed37-fit
  float32 precursor is preserved separately and never pooled into this batch.
- Limitations: non-independent research data, missing outcomes, selected-tail
  underprediction, trainable-capacity difference and bounded adaptation range.

Observation8/prediction12 at raw-frame stride12; dataset-local coordinates.
No metric/time calibration, human-gold, true3D, foundation or physical-safety
claim. Stage5C and SMC remain disabled. Deployment is unchanged.
