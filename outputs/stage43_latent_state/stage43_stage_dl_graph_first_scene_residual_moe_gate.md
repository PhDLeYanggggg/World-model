# Stage43-DL Graph-First Scene-Residual MoE Gate

- verdict: `stage43_dl_graph_first_scene_residual_moe_pass_safe_bq_lift_diagnostic`
- passed: `15 / 15`
- graph-first MoE executed: `True`
- beats best single: `False`
- beats BQ gated fusion: `True`
- safe easy: `True`
- deployable policy changed: `False`
- long objective complete: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | passed |
| --- | --- |
| `dk_next_training_contract_present` | `True` |
| `fresh_graph_first_moe_trained` | `True` |
| `graph_default_and_scene_residual_architecture` | `True` |
| `expert_preservation_loss_recorded` | `True` |
| `scene_and_graph_dims_present` | `True` |
| `latent_noncollapse` | `True` |
| `protected_eval_completed` | `True` |
| `easy_preservation_measured` | `True` |
| `best_single_comparison_reported` | `True` |
| `bq_comparison_reported` | `True` |
| `scene_residual_gate_measured` | `True` |
| `checkpoints_not_committed` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
