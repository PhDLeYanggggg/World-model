# Stage43-BX Latent Risk Head Robustness Gate

- verdict: `stage43_bx_latent_risk_head_robustness_pass_horizon_caveat`
- passed: `12 / 12`
- horizon min AUROC: `0.6147`
- weak horizon slices: `5`
- deployable policy changed: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | passed |
| --- | --- |
| `stage43_m_checkpoint_replayed` | `True` |
| `stage43_y_precondition_seen` | `True` |
| `fresh_test_predictions_completed` | `True` |
| `latent_noncollapse` | `True` |
| `global_failure_gain_harm_heads_strong` | `True` |
| `per_domain_heads_robust` | `True` |
| `per_horizon_heads_supported` | `True` |
| `bootstrap_ci_completed` | `True` |
| `weak_horizon_caveats_reported` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
