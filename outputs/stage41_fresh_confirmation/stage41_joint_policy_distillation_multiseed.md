# Stage41 Joint Distiller Multi-Seed Replication

- source: `fresh_run`
- seeds: `[11, 17, 23]`
- replication pass: `True`
- metric summary: `{'all_improvement': {'mean': 0.2855577482364627, 'std': 0.004928586535177756, 'min': 0.2785936928672702, 'max': 0.2892905695876844}, 't50_improvement': {'mean': 0.19436183988319766, 'std': 0.017537607009276843, 'min': 0.1695617463193938, 'max': 0.20702267170624478}, 't100_improvement': {'mean': 0.2995344632130104, 'std': 0.0017860040352511645, 'min': 0.29713360387900334, 'max': 0.30141432018637826}, 'hard_failure_improvement': {'mean': 0.2862145627536274, 'std': 0.005438774247628364, 'min': 0.27852302033504284, 'max': 0.29008334249034573}, 'easy_degradation': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}, 'switch_rate': {'mean': 0.4033100417807233, 'std': 0.015237735065923564, 'min': 0.38178936752629306, 'max': 0.41503385679296934}}`
- positive domain counts: `[2, 2, 2]`
- no leakage: `{'base_switch_input': False, 'future_waypoints_input': False, 'future_labels_eval_only': True, 'test_threshold_tuning': False, 'central_velocity': False, 'test_endpoint_goals': False}`

UCY remains fallback-only in the main candidate; this replication checks seed stability, not UCY repair.
