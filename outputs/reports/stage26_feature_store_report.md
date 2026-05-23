# Stage 26 SDD Causal Feature Store Report

- 当前不是 true 3D / foundation model；SDD 仍是 pixel-space raw-frame benchmark。
- Feature store is built from Stage24 medium baseline-evaluated rows, not unevaluated windows.
- All features come from causal past or current start-frame state; baseline errors are labels only.

- features: `57`
- build time seconds: `64.825`
- leakage audit: `{'future_endpoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'baseline_errors_as_features': False, 'velocity_source': 'causal finite difference from Stage21/24 fast cache'}`

| split | rows | features | hard candidates | finite feature fraction |
| --- | ---: | ---: | ---: | ---: |
| train | 40000 | 57 | 39551 | 1.000000 |
| val | 20000 | 57 | 19802 | 1.000000 |
| test | 100000 | 57 | 96581 | 1.000000 |
