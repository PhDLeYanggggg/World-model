# Frozen Appearance Predictor: Training-Support Control

## Material Passport

Mode: run, exploratory fit-only failure diagnosis. The preceding18-model result
is negative. This follow-up tests whether measured camera/geometry covariate
shift can explain forecast harm under a fixed predictor. It is not new neural
training, independent confirmation or a change to the approved parent protocol.

## Fixed Intervention

Retain all18 completed checkpoints, original365 rows, folds, three seeds, input
arms, image masks, labels, 0.9 diagnostic gate and exact ADE/FDE definitions.
Do not choose any setting from its held result. No calibration/development/final
test rows or labels are opened. Use the same native-index assumption and supplied
coordinate H; neither physical seconds nor metric scale is newly verified.

Compute a marginal minimum/maximum box from each checkpoint's training rows
only, before its original mean/std normalization. Report all four settings:

1. Original: exact replay, no changed input or gate.
2. Jacobian box: clamp only four camera-Jacobian features (columns28:32) to the
   training minima/maxima. Recompute network output/probability, same0.9 gate.
3. All-feature box: clamp all32 causal geometry features to training minima/
   maxima. Images/masks, output H and labels remain unchanged. Same0.9 gate.
4. Support fallback: leave original network input/prediction unchanged, but use
   exactzero/CV whenever any of32 raw geometry features lies outside the training
   box. Combine with the original0.9 and complete-image gate. This is abstention,
   not better forecasting; report retained support and switch coverage explicitly.

Training features are unchanged by box projection. Constant training features
stay constant. No held minima/maxima, future targets, error-defined subsets or
held probability distributions determine the treatment. This rectangle is NOT
joint-distribution support, a causal intervention or a formal safety guarantee.
Clipping can destroy correlations or hide legitimate unseen scene geometry.

## Execution and Interpretation

Native arm64 CPU4/interop1/workers0; no optimizer or refitting. Save every local
row prediction with hashes, per-trial receipts, PID/heartbeat and completion.
Original arm must reproduce all saved outputs within1e-10 before accepting any
treated scores. Resume revalidates completed receipts and does not silently
repeat evaluations. Report all72 predictor/treatment combinations and seed
means, unrestricted/gated native and parent ADE/FDE, start Brier, easy absolute
harm, original-vs-treated differences, changed-input coverage and switch rate.

If harm falls but no positive gain appears, conclude support treatment reduces
damage only. If fallback rejects nearly all queries, report missing support,
not successful generalization. Positive treated gains, if any, remain adaptive
fit-only evidence requiring a new independently designed experiment. No winner
promotion, hidden seed filtering or formal CI from two exposed scenes.

The primary eight-observed/twelve-predicted native-step task, raw50 supplement
and pending prospective metric decision are unchanged. Original negatives stay
immutable. No deployment, Stage5C execution, SMC or submission-readiness claim.
