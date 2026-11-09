# Nested causal spline choice: corrected runtime

This run follows the [initial design](../opendd_nested_spline_choice_v1/protocol.md).
The first process stopped before calibration or held scoring: an
inference-mode scope accidentally covered construction of a nested
training model and prevented backpropagation. The corrected runner
constructs or restores the model before entering its prediction-only
function. It starts from a separate checkpoint directory; the aborted
run contributes no fitted model, score or selection to this result.
The forecast bank, split roles, controls and scientific rules are unchanged.
