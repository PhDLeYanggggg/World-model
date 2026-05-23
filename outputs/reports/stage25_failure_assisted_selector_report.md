# Stage 25 Failure-assisted Selector Report

- Uses a causal failure-risk signal to decide when selector switches are allowed.
- Stage24 failure AUROC: `0.8714731936246393`
- Stage25 failure proxy AUROC: `0.8721049504602977`
- selected policy: `{'policy_family': 'failure_assisted_regret', 'confidence_threshold': 0.35, 'predicted_gain_threshold_px': 0.0, 'min_predicted_margin_px': 0.0, 'easy_guard': True, 'easy_predicted_strongest_threshold_px': 10.0, 'failure_probability_threshold': 0.1}`
- t+50 improvement: `0.010378353851759892`
- hard/failure improvement: `0.007118169738036806`
- easy degradation: `0.009405263004718556`
- harm over fallback: `-0.18313562636613845`
- passed gate: `False`
