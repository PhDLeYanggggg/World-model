# Stage43-DJ Latent World-State Current Reconciliation

- source: `fresh_stage43_dj_latent_world_state_current_reconciliation`
- result_source: `fresh_reconciliation_from_stage43_c_y_bh_di`
- verdict: `stage43_dj_latent_world_state_current_reconciliation_pass`
- gate: `13 / 13`
- protected multimodal latent-state candidate: `True`
- standalone world model deployable: `False`
- t100 support-aware head deployed: `False`

## Current Read

M3W currently has a protected dataset-local/raw-frame multimodal latent-state candidate. The strongest honest claim is protected world-state evidence with useful proxy heads, not a standalone ungated true-3D or foundation model.

## Protected Latent-State Metrics

- all improvement: `17.77%`
- t50 improvement: `13.75%`
- t100 raw-frame diagnostic: `1.82%`
- hard/failure improvement: `18.16%`
- easy degradation: `0.00%`
- switch rate: `17.65%`

## Ungated Neural Diagnostic

- all improvement: `59.86%`
- t50 improvement: `63.35%`
- t100 raw-frame diagnostic: `47.56%`
- hard/failure improvement: `63.84%`
- easy degradation: `1.41%`
- This remains diagnostic, not a reason to drop the safety floor.

## Proxy Head Suite

- failure risk AUROC: `0.8648`
- gain opportunity AUROC: `0.8737`
- harm guard AUROC: `0.9047`
- causal history-density R2: `0.8178`
- future interaction-risk AUROC: `0.7694`
- deployable proxy heads: `['failure_risk', 'gain_opportunity', 'harm_guard', 'causal_history_density', 'future_interaction_risk']`
- diagnostic-only heads: `['waypoint_label_availability', 'smoothness_validity_proxy']`

## T100 Support-Aware Head

- mean t100: `0.16%`
- mean min-without-group t100: `0.08%`
- all min-without-group positive: `True`
- max easy degradation: `0.00%`
- beats DH t100 mean: `False`
- beats DE t100 mean: `False`
- deployed: `False`

## Boundary

- Dataset-local/raw-frame 2.5D only.
- No metric or seconds-level claim.
- No true-3D or foundation claim.
- Safety floor remains required.
- Stage5C remains false.
- SMC remains false.

## Gate

| gate | passed |
| --- | --- |
| `protected_latent_eval_passed` | `True` |
| `protected_latent_metrics_positive_easy_safe` | `True` |
| `multimodal_head_suite_passed` | `True` |
| `latent_noncollapse` | `True` |
| `proxy_heads_strong_enough` | `True` |
| `protected_candidate_lock_passed` | `True` |
| `safety_floor_required_not_hidden` | `True` |
| `t100_support_head_passed_but_diagnostic` | `True` |
| `t100_support_head_safe_no_deployment_lift` | `True` |
| `no_future_or_test_leakage` | `True` |
| `claim_boundary_not_overstated` | `True` |
| `stage5c_and_smc_false` | `True` |
| `long_objective_kept_active` | `True` |
