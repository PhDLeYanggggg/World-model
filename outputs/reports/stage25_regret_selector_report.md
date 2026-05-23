# Stage 25 Regret Selector Report

- 任务从 hard best-baseline classification 改为 per-baseline expected FDE prediction。
- baseline errors 只作为 regression labels/evaluation targets，不作为 inference feature。
- 使用 confidence/gain/easy guard fallback 到 global strongest baseline。

- best model: `random_forest`
- best policy: `{'policy_family': 'regret_random_forest', 'confidence_threshold': 0.1, 'predicted_gain_threshold_px': 5.0, 'min_predicted_margin_px': 0.0, 'easy_guard': True, 'easy_predicted_strongest_threshold_px': 10.0}`
- expected-FDE log RMSE: `1.9290820620018254`
- ranking accuracy: `0.14936`
- selected baseline FDE improvement: `0.010836345024840321`
- t+50 improvement: `0.013572603215101897`
- hard/failure improvement: `0.010929538646649695`
- easy degradation: `0.00965540295110423`
- harm over fallback: `-0.28119418047189715`
- selector regret: `6.562430680690706`
- passed gate: `False`
