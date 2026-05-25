# Stage41 Fixed-Composer Residual Source Oracle Audit

- source: `fresh_run`
- oracle is diagnostic, not deployable: `True`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- headroom domains: `[]`
- two-domain residual oracle headroom: `False`

| domain | fixed policy | oracle delta all/t50/t100/hard | oracle switch | positive residual | hard switch | source distribution |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `ETH_UCY` | `gain_gate/bridge/old_shape` | 0.000086/0.000013/0.000000/0.000073 | 0.760533 | 0.001111 | 0.726319 | `{'bridge': 0.9952773404944902, 'old_shape': 4.630058338735068e-05, 'gain_gate': 0.004676358922122419}` |
| `TrajNet` | `bridge/gain_gate/gain_gate` | 0.000244/0.000109/0.000708/0.000268 | 0.359714 | 0.001374 | 0.420090 | `{'bridge': 0.998351195383347, 'old_shape': 0.0010992030777686177, 'gain_gate': 0.0005496015388843089}` |

## Interpretation

- This audit asks whether any per-row oracle over bridge / old-shape / gain-gate can beat the validation-selected fixed composer.
- Future waypoint labels are used only for diagnostic oracle costs; they are not model inputs, thresholds, goals, or deployment features.
- If oracle headroom is small, more source-switch learners are unlikely to help. If oracle headroom is large but learned switches fail, the blocker is causal feature separability.
- no leakage: `{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'future_waypoints_input': False, 'future_waypoints_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'oracle_future_labels_diagnostic_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
- claim boundary: `{'oracle_diagnostic': True, 'deployable_model': False, 'latent_generative_rollout': False, 'metric_or_seconds_claim': False, 'true_3d': False, 'foundation_world_model': False}`
