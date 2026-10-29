# Inference-Only Singleton-Axis Repair

The first readout PID60443 exits1before creating any result group. In the
neural-only comparator, NumPy interprets a one-element Torch index as a scalar,
collapsing the candidate axis. The model correctly rejects the resulting input
shape. This is not an OpenMP freeze, failed training or scientific gate result.

Keep the registered training script/API,54frozen checkpoints,optimizer states,
all data,roles,features,decision rule and thresholds unchanged. A separate
inference adapter uses an explicit NumPy integer array for NumPy selection and
a Torch tensor only within Torch. Multi-action outputs match the frozen
function exactly; a singleton regression test checks the intended tensor shape
and legal neural/floor decisions. No NumPy predictor or replacement model is
introduced: the original trained Torch model still performs the numerical work.

Commit this adapter and amendment before the corrected first scientific
readout. Preserve the failed attempt and its exit status. Use the new evaluation
entry for both first readout and exact replay; do not rewrite training bindings
or retrain to conceal an inference indexing error.
