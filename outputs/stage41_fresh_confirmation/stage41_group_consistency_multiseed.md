# Stage41 Group Consistency Distiller Multi-Seed Replication

- source: `fresh_run`
- seeds: `[11, 17, 23]`
- replication pass: `False`
- metric summary: `{'all_improvement': {'mean': 0.2065531520646495, 'std': 0.02835578310041668, 'min': 0.17701361035196395, 'max': 0.24480988226889844}, 't50_improvement': {'mean': 0.16263377672309107, 'std': 0.014247531031898833, 'min': 0.15038632705695498, 'max': 0.18261348197127147}, 't100_improvement': {'mean': 0.20528445525217295, 'std': 0.04323068746473773, 'min': 0.15169194522133755, 'max': 0.25756051912591715}, 'hard_failure_improvement': {'mean': 0.20900444143596122, 'std': 0.028900501258207947, 'min': 0.1785201809464737, 'max': 0.24782390944963728}, 'easy_degradation': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}, 'switch_rate': {'mean': 0.25414205445901167, 'std': 0.026757447732981676, 'min': 0.2304783172453537, 'max': 0.2915466071171301}, 'collision_delta_vs_floor_005': {'mean': 0.009587047702899329, 'std': 0.0013939009080835302, 'min': 0.00808907149932131, 'max': 0.011445761033053281}}`
- positive domain counts: `[3, 3, 3]`
- no leakage: `{'future_waypoints_input': False, 'future_labels_eval_only': True, 'train_gain_safe_unsafe_labels_only': True, 'test_threshold_tuning': False, 'policy_selected_on_val': True, 'central_velocity': False, 'test_endpoint_goals': False, 'stage5c_executed': False, 'smc_enabled': False}`

All policies are selected on validation and evaluated on test once. Coordinates remain dataset-local/raw-frame; this is not true 3D, foundation-scale, Stage5C, or SMC.
