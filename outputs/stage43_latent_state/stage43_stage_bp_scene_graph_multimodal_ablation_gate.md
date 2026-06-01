# Stage43-BP Scene-Graph Multimodal Ablation Gate

- verdict: `stage43_bp_scene_graph_multimodal_ablation_pass_negative_unsafe_diagnostic`
- passed: `16 / 16`
- multimodal ablation executed: `True`
- multimodal contribution supported: `True`
- best-single lift supported: `False`
- full multimodal unsafe: `True`
- deployable policy changed: `False`
- long objective complete: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | passed |
| --- | --- |
| `scene_proxy_cache_precondition_passed` | `True` |
| `scene_proxy_retrained_ablation_precondition_passed` | `True` |
| `graph_history_retrained_ablation_precondition_passed` | `True` |
| `fresh_retrained_multimodal_variants` | `True` |
| `no_scene_no_graph_baseline_retrained` | `True` |
| `scene_only_and_graph_only_retrained` | `True` |
| `full_scene_graph_uses_both_modalities` | `True` |
| `bootstrap_recorded` | `True` |
| `latent_noncollapse` | `True` |
| `multimodal_beats_no_context_on_t50_or_hard` | `True` |
| `multimodal_best_single_comparison_reported` | `True` |
| `easy_safety_measured` | `True` |
| `checkpoints_not_committed` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
