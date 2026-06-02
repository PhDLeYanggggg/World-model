# Stage43 Current World-Model Gate

- source: `fresh_stage43_ca_latent_adapter_downstream_heads`
- verdict: `stage43_ca_latent_adapter_downstream_heads_partial_lift`
- passed: `13 / 16`
- protected multimodal latent state candidate: `True`
- selected adapter variant: `identity_stage43m_adapter_z`
- adapter downstream mean ADE: `0.2961`
- adapter risk mean AUROC: `0.8910`
- protected all improvement: `0.0324`
- protected t50 improvement: `-0.0022`
- deployable policy changed: `False`
- long objective complete: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

## Current Boundary

- Stage43-CA is a train-only downstream head audit, not an ungated deployment policy.
- Safety floors remain required for deployment.
- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.

| gate | passed |
| --- | --- |
| `stage43_bz_precondition_passed` | `True` |
| `train_only_downstream_heads_fit` | `True` |
| `future_labels_eval_only` | `True` |
| `no_test_threshold_tuning` | `True` |
| `selected_variant_contains_adapter` | `True` |
| `adapter_variant_validation_selected` | `True` |
| `adapter_waypoint_ungated_beats_identity` | `True` |
| `adapter_waypoint_ungated_beats_stage43_m` | `False` |
| `adapter_risk_auc_beats_identity` | `False` |
| `adapter_risk_auc_beats_stage43_m` | `True` |
| `protected_eval_completed` | `True` |
| `protected_easy_preserved` | `False` |
| `protected_adapter_lift_vs_floor` | `True` |
| `domain_and_horizon_breakdowns_reported` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
