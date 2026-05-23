# Stage 26 Failure-assisted Selector Report

- Failure probability low blocks switching; high failure probability allows switching.
- Uses the Stage24-passed failure-predictor role as auxiliary gating, retrained on Stage26 causal features.

- Stage24 failure AUROC: `0.8714731936246393`
- Stage26 failure aux AUROC: `0.948231574450493`
- selected policy: `{'policy_family': 'stage26_failure_assisted_expected_fde', 'risk_weight': 0.0, 'confidence_threshold': 0.0, 'predicted_gain_threshold_px': 10.0, 'easy_predicted_threshold_px': 10.0, 'max_switch_risk_delta': 0.05, 'failure_probability_threshold': 0.2}`
- t+50 improvement: `0.14583655843823773`
- hard/failure improvement: `0.11232167634621226`
- easy degradation: `0.01808836280803794`
- passed gate: `True`
