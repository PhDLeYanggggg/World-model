# Frozen-Model Constant-Feature Diagnostic

Registered while the full visual study runs, after inspecting input support and
the first seed's completed held fit predictions. This is adaptive fit-only
diagnosis, not a prespecified primary claim or a new test set.

ETH has different raw annotation-step metadata from Hotel/Zara. In its held-fit
fold, raw horizon and raw frame-step features are constant in training and differ
on every held row. Train-standardization therefore clips both to -10. An
untrained feature direction can perturb the network despite unchanged task steps.
This supports a concrete implementation/support hypothesis, not proof of the
cause of all forecasting errors.

For all 36 frozen models, set exactly constant training dimensions to their
training value before the original standardization. No epsilon search. Every
training row must remain bitwise identical. Variable dimensions and all labels,
weights, thresholds and predictions in the main report remain unchanged.
Recompute original predictions first and require exact replay, then evaluate the
fixed diagnostic. Report all seeds/folds and easy harm; do not pick favorable
arms or call reduced damage predictive success. No retraining or deployment.
