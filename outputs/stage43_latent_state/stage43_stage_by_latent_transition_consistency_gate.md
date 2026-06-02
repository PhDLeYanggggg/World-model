# Stage43-BY Latent Transition Consistency Gate

- verdict: `stage43_by_latent_transition_consistency_pass_with_readout_caveat`
- passed: `13 / 13`
- global transition gain vs identity: `0.7450`
- global transition gain vs train centroid: `-0.0357`
- calibrated readout gain vs identity: `-0.0177`
- calibrated readout gain vs train centroid: `0.3097`
- weak transition slices: `4`
- calibrated weak transition slices: `5`
- deployable policy changed: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | passed |
| --- | --- |
| `stage43_m_checkpoint_replayed` | `True` |
| `stage43_bx_precondition_seen` | `True` |
| `fresh_transition_predictions_completed` | `True` |
| `future_target_latent_label_eval_only` | `True` |
| `latent_noncollapse` | `True` |
| `raw_transition_lift_vs_identity` | `True` |
| `calibrated_readout_lift_vs_train_centroid` | `True` |
| `bootstrap_transition_lift_supported` | `True` |
| `domain_and_horizon_breakdowns_reported` | `True` |
| `raw_centroid_and_identity_readout_caveats_reported` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
