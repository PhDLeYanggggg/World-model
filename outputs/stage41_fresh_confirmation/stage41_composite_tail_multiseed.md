# Stage41 Composite-Tail Multi-Seed Evidence

- source: `fresh_run`
- seed source: `cached_verified_teacher_guided_multiseed_checkpoints_re_evaluated_fresh`
- seeds: `[11, 17, 23]`
- replication pass: `True`
- strict delta vs teacher repair pass: `True`
- metric summary: `{'all_improvement': {'mean': 0.20954401723273208, 'std': 0.000812132451000419, 'min': 0.20858725214166063, 'max': 0.2105726625312615}, 't50_improvement': {'mean': 0.1383020020634588, 'std': 0.0008215631373107549, 'min': 0.13750499067156252, 'max': 0.13943264750957252}, 't100_improvement': {'mean': 0.1445226429961963, 'std': 0.0024185410870793955, 'min': 0.14199993286084633, 'max': 0.1477842384655702}, 'hard_failure_improvement': {'mean': 0.203088119625216, 'std': 0.0009137514198866433, 'min': 0.2019696303499685, 'max': 0.2042078540414125}, 'easy_degradation': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}, 'alpha_mean': {'mean': 0.3022868942995726, 'std': 0.001235759500323106, 'min': 0.3010394755798877, 'max': 0.30397060942227344}, 'switch_rate': {'mean': 0.3427520049944772, 'std': 0.0037883401613774035, 'min': 0.339576429909235, 'max': 0.34807664601642413}, 'collision_delta_vs_floor_005': {'mean': -0.003961994203749227, 'std': 0.0001958450099422415, 'min': -0.004127077295572101, 'max': -0.003686855717377757}}`
- delta vs teacher repair summary: `{'all_delta': {'mean': 0.005549847936104062, 'std': 0.0007554235504385845, 'min': 0.004997329897068581, 'max': 0.006617965511956925}, 't50_delta': {'mean': 0.006532821969673974, 'std': 0.0009042920713246326, 'min': 0.005636676602763568, 'max': 0.00777102681701991}, 't100_delta': {'mean': 0.009578016235438488, 'std': 0.0014294941686291741, 'min': 0.008330490182942518, 'max': 0.011579436486280126}, 'hard_delta': {'mean': 0.006046163872124384, 'std': 0.0008354645702825151, 'min': 0.0054317016626029835, 'max': 0.0072273679267517155}, 'easy_delta': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}}`
- positive domain counts: `[3, 3, 3]`
- no leakage: `{'future_endpoint_input': False, 'future_labels_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'policy_selected_on_val': True, 'stage5c_executed': False, 'smc_enabled': False}`

Composite-tail is selected on validation for each seed-specific checkpoint and evaluated on test once. It is not Stage5C or SMC.
