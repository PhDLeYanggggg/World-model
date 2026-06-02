# Stage43-CA Latent Adapter Downstream Heads

- source: `fresh_stage43_ca_latent_adapter_downstream_heads`
- result_source: `fresh_train_only_downstream_head_audit`
- verdict: `stage43_ca_latent_adapter_downstream_heads_partial_lift`
- gate: `13 / 16`
- selected adapter variant: `identity_stage43m_adapter_z`
- best overall validation variant: `identity_stage43m_adapter_z`
- deployable policy changed: `False`

## Variant Comparison

| variant | mean ADE | mean FDE | risk mean AUROC | protected all | protected t50 | hard/failure | easy degradation | switch rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `identity_z_t` | `0.3173` | `0.1804` | `0.8932` | `0.0407` | `-0.0012` | `0.0725` | `0.0284` | `0.1835` |
| `stage43_m_z_next` | `0.2939` | `0.1762` | `0.8897` | `0.0586` | `-0.0018` | `0.0938` | `0.0331` | `0.1635` |
| `stage43_bz_adapter_z_next` | `0.3256` | `0.1872` | `0.8790` | `-0.0093` | `-0.0019` | `0.0268` | `0.0735` | `0.1743` |
| `identity_plus_adapter_z` | `0.2977` | `0.1720` | `0.8934` | `0.0183` | `-0.0003` | `0.0430` | `0.0339` | `0.1429` |
| `stage43_m_plus_adapter_z` | `0.3056` | `0.1627` | `0.8898` | `0.0288` | `-0.0014` | `0.0577` | `0.0349` | `0.1602` |
| `identity_stage43m_adapter_z` | `0.2961` | `0.1539` | `0.8910` | `0.0324` | `-0.0022` | `0.0587` | `0.0427` | `0.1166` |

## Interpretation

- Stage43-CA fits identical train-only downstream heads on identity `z_t`, Stage43-M `z_next`, Stage43-BZ adapter `z_next`, and current+future-latent concatenations.
- Future waypoint/risk/density labels are used only for train/eval targets, not as inference inputs.
- Validation selects the protected safe-switch policy; test is evaluated once.
- This is downstream/world-state evidence for the latent adapter, not a deployment change and not a safety-floor removal.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.

## Gate

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
