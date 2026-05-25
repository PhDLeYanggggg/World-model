# Stage41 Teacher-Guided Proposal Multi-Seed Replication

- source: `fresh_run`
- seeds: `[11, 17, 23]`
- validation collision ceiling: `0.007`
- test collision ceiling: `0.01`
- replication pass: `True`
- metric summary: `{'all_improvement': {'mean': 0.20399416929662803, 'std': 0.00034730418128016015, 'min': 0.20358992224459205, 'max': 0.20443788862598744}, 't50_improvement': {'mean': 0.13176918009378483, 'std': 0.0015039436795714375, 'min': 0.13019734119222148, 'max': 0.13379597090680895}, 't100_improvement': {'mean': 0.1349446267607578, 'std': 0.0010351105082027497, 'min': 0.13366944267790382, 'max': 0.13620480197929008}, 'hard_failure_improvement': {'mean': 0.1970419557530916, 'std': 0.0004387893655068093, 'min': 0.19653792868736553, 'max': 0.19760745245724853}, 'easy_degradation': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}, 'switch_rate': {'mean': 0.2987681890217548, 'std': 0.001572253031675338, 'min': 0.29694928684627575, 'max': 0.30078518945396915}, 'collision_delta_vs_floor_005': {'mean': -0.004072049598297822, 'std': 0.00023394263019810041, 'min': -0.004255475255878827, 'max': -0.0037418834146520363}}`
- positive domain counts: `[3, 3, 3]`
- no leakage: `{'teacher_switch_inference_input': False, 'teacher_switch_train_label_only': True, 'future_waypoints_input': False, 'future_waypoints_label_eval_only': True, 'future_endpoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'policy_selected_on_val': True, 'proximity_guard_selected_on_val': True, 'stage5c_executed': False, 'smc_enabled': False}`

Each seed is trained fresh, selects policy and proximity guard on validation, and evaluates test once. Future waypoints are labels/eval only.
