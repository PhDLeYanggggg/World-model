# Stage 26 Expected-FDE Selector Report

- Trains per-baseline expected FDE/risk prediction from causal SDD features.
- Does not train residuals, latent generative models, JEPA, or SMC.

- best model: `ridge`
- selected policy: `{'policy_family': 'stage26_expected_fde_ridge', 'risk_weight': 0.0, 'confidence_threshold': 0.0, 'predicted_gain_threshold_px': 10.0, 'easy_predicted_threshold_px': 10.0, 'max_switch_risk_delta': 0.05}`
- expected FDE log RMSE: `0.9627713308405523`
- ranking accuracy: `0.40484`
- t+50 improvement: `0.14569782621903304`
- hard/failure improvement: `0.11539720290780353`
- easy degradation: `0.02429346208315719`
- selector regret: `3.872658875556886`
- passed gate: `True`
