# M3W-Neural v1

M3W-Neural v1 is a Stage41-protected neural world-dynamics candidate. It combines composite-tail bounded neural dynamics with a validation-selected safe-switch policy and the Stage37/teacher safety floor.

It is not true 3D, not metric, not seconds-level, not a foundation model, and not Stage5C/SMC.

## Files

- `report_m3w_neural_v1.md` — frozen result summary.
- `evidence_matrix_m3w_neural_v1.md/json` — gate and metric evidence.
- `selector_policy_m3w_neural_v1.json` — frozen policy metadata and hashes.
- `model_card_m3w_neural_v1.md` — intended use and limitations.
- `data_card_m3w_neural_v1.md` — dataset and leakage status.
- `reproducibility_m3w_neural_v1.md` — rerun commands.
- `paper_gap_m3w_neural_v1.md` — what is still missing before stronger publication claims.

Latest package inputs include the negative fixed-composer source-switch audits and the negative strict pure-UCY neural retrain audit, so the frozen package records both the successful composite-tail path and the exhausted source-switch / source-only retrain branches.

The package also includes the positive endpoint-to-full bridge audit: domain-local endpoint neural dynamics pass actual full-waypoint ADE/FDE, multi-agent, proximity, and smoothness gates on ETH_UCY and TrajNet through a linear waypoint bridge. This strengthens world-state evidence without claiming learned waypoint-shape dynamics.

The package includes a calibrated learned-shape meta-policy as well. It selects protected waypoint-shape residual sources on validation, evaluates test once, and remains positive on ETH_UCY and TrajNet. The learned-shape contribution is small and protected, not an ungated neural replacement.
