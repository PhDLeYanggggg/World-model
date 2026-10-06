# Avoiding Duplicate Repair Claims

These existing experiments constrain the next hypothesis. They are cached
development reports, not results from the present TRAIN-only replay. Their
different populations and estimands must not be pooled numerically.

| Earlier control | What it tested | Why it does not solve the present problem |
|---|---|---|
| Native underharm4 | Fourfold penalty for underestimating raw harm on SDD | Fewer harmful interventions also meant fewer useful interventions; strict protection still failed. An expectile is not a calibrated mean or confidence bound. |
| Selected-group risk learning | Symmetric group-mean moment error on fixed causal subgroups | Lower fitting-group loss did not beat the matched mean-loss policy. Useful joint allocation did not establish safety or an advantage over the strongest old rule. |
| Easy-risk gradient priority | Capped auxiliary gradients relative to direct-risk gradients | A small gain over the uncapped arm did not beat the raw-risk control or repair risk. More epochs or another generic weight sweep is not justified by this result. |
| Selected-set recalibration | Refit margins on the retained source-OOF population | Complete support did not improve and conservative utility fell. Fewer violations on a smaller population were not a repair. |
| Current temporal auxiliary | Temporal versus row-mean/no auxiliary supervision, matched 216 fits | Lower signed MSE versus row-mean accompanied worse decisions and failed easy-risk control. |

The present component decomposition has a narrower purpose: test whether the
frozen heads already misjudge their own training-set selected population, and
whether the observed numerical error comes mainly from harm, reference or both.
It cannot by itself prove that another asymmetric loss will generalize. Any
successor needs a single explicit changed factor, the same source roles and
fixed budgets, coverage-matched controls, unknown-outcome accounting, and all
negative results retained. TRAIN improvement alone cannot open transfer or
independent confirmation.

Sources:
- [Native gain/harm](../../native_gain_harm_v1/conclusions.md)
- [Selected risk learning](../../european_selected_risk_learning_v1/conclusions.md)
- [Gradient priority](../../european_easy_risk_priority_v1/conclusions.md)
- [Selected-set calibration](../../european_selected_set_calibration_v1/conclusions.md)
- [Current fixed readout](../readout/failure_analysis.md)

Image-local detector-silver and raw annotation-frame evidence only. No metric,
seconds, true-3D, foundation or deployment claim. Stage5C and SMC remain off.
