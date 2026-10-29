# Existing Controls and the Remaining Distinction

The preceding goal turn made progress:72 fitting views and288 disposable
gradient probes isolated a loss-versus-risk discrepancy. This round executes
a matched objective intervention, rather than another clipping diagnosis.

| Prior evidence | What was tested | Why not silently rerun it |
|---|---|---|
| [Fixed-floor tail](../european_fixed_floor_tail_v1/conclusions.md) | Nonnegative bounded moment heads with ordinary versus tail-weighted loss | Tail weighting did not establish safe matched utility; do not add another weight sweep. |
| [Hurdle risk](../european_hurdle_risk_v1/conclusions.md) | Occurrence/severity supervision | Ranking and mean prediction can diverge; not a new hurdle trial. |
| [Selected-risk learning](../european_selected_risk_learning_v1/conclusions.md) | Selected-group versus mean objectives and joint decisions | Added selected loss did not improve accuracy; no new selected-group weighting here. |
| [Decoder/risk factorial](../european_decoder_risk_factorial_v1/readout/conclusions.md) | Shared/source decoders with/without risk penalty | Benefit and reference outputs were also trainable; not the same protected-estimator intervention. |
| [OOF magnitude](../european_oof_magnitude_v1/protocol.md) | Honest fitted scalar magnitude readout | Does not isolate gradient interference in the current frozen-utility neural head. |
| [Query conditioning](../european_query_conditioning_v2/conclusions.md) | Same-cohort errors/Jacobians and exact-state one-step clipping counterfactuals | Total objective improves more consistently than selected easy-risk MSE; no persistent training tested. |

The new contrast starts both arms from each of18 identical final self heads,
including AdamW moments. It protects B/R/R_easy, uses identical additional
query draws and budget, and changes only whether the risk estimator receives
the regret/dual action losses. The estimation arm retains the original five
moment/signed-score quadratic objective; the other continues the full original
objective. This is a local continuation test, not a from-scratch comparison or
a claim that decoupling itself is a novel method.

Decision-focused learning is established literature. [Smart Predict, then
Optimize, arXiv1710.08005v5](https://arxiv.org/pdf/1710.08005) explicitly separates
prediction error from induced decision error and derives SPO+ for its setting.
Read scope: introduction and formulation, not a complete theorem audit. Our
soft intervention loss is not SPO+; its selected-risk setting and unknown
outcome support are not certified by that paper's consistency results. The
current test asks whether a particular decision loss harms conditional cost
estimation under protected outputs, not whether all decision-focused learning
is inferior to regression.

Existing source evidence remains exploratory. No historical Stage37 score is
recertified and no new forecasting or world-dynamics claim follows from heads.
