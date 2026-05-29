# Stage42-JY Gate

- source: `fresh_stage42_jy_context_materiality_by_source_slice`
- passed: `14 / 14`
- verdict: `stage42_jy_context_materiality_by_source_slice_pass`

| gate | pass |
| --- | ---: |
| `ao_incremental_ablation_loaded` | `True` |
| `jt_claim_refresh_passed` | `True` |
| `jv_source_slice_matrix_passed` | `True` |
| `js_context_closure_passed` | `True` |
| `baseline_family_control_positive` | `True` |
| `standalone_context_signal_recorded` | `True` |
| `no_material_global_incremental_context` | `True` |
| `narrow_slice_signals_recorded` | `True` |
| `blocked_context_claim_preserved` | `True` |
| `next_training_spec_emitted` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_3d_foundation` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |
