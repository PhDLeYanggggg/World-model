# Stage41 Group Consistency Multi-Seed Safety-Buffer Repair

- source: `fresh_run`
- prior replication pass: `False`
- validation collision ceiling: `0.005`
- test collision ceiling: `0.01`
- replication pass after repair: `True`
- metric summary: `{'all_improvement': {'mean': 0.139879916889688, 'std': 0.01431910434689191, 'min': 0.11979149779858422, 'max': 0.15213697251233183}, 't50_improvement': {'mean': 0.12149802173622119, 'std': 0.005174434446659948, 'min': 0.11587051220062827, 'max': 0.12836276528260981}, 't100_improvement': {'mean': 0.1689208799266194, 'std': 0.02465489382312882, 'min': 0.13813952571953958, 'max': 0.1984952398744826}, 'hard_failure_improvement': {'mean': 0.14504228036236014, 'std': 0.014274616476767789, 'min': 0.12488209249854709, 'max': 0.15602908941492988}, 'easy_degradation': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}, 'switch_rate': {'mean': 0.15134706814580032, 'std': 0.012378867345897106, 'min': 0.1344546895260049, 'max': 0.16377323152283532}, 'collision_delta_vs_floor_005': {'mean': 0.00704354525110971, 'std': 0.000901961983428898, 'min': 0.0062731574892695985, 'max': 0.008309182288418482}}`
- positive domain counts: `[3, 3, 3]`
- no leakage: `{'future_waypoints_input': False, 'future_labels_eval_only': True, 'train_gain_safe_unsafe_labels_only': True, 'test_threshold_tuning': False, 'policy_selected_on_val': True, 'central_velocity': False, 'test_endpoint_goals': False, 'stage5c_executed': False, 'smc_enabled': False}`

This is a validation-selected conservative deployment repair over already-trained seed checkpoints. Test thresholds are not tuned on test; future labels remain label/eval only.
