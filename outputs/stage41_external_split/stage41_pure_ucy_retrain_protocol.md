# Stage41 Pure UCY Train/Val/Test Policy-Head Calibration

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- protocol: `pure_ucy_train_val_test_policy_head_calibration_over_frozen_neural_proposal`
- train/val/test rows: `117808` / `16103` / `35441`
- selected policy: `{'type': 'pure_ucy_ridge_gain_harm', 'mode': 'teacher_raw_gain_harm', 'alpha': 0.4, 'gain_min': 0.0, 'harm_max': 0.0, 'proposal_harm_max': 0.25, 'uncertainty_max': 0.25, 'teacher_min': 0.5}`
- pure UCY policy train/val/test gate: `True`
- strict pure UCY-only neural retrain/select/test gate: `False`
- remaining blocker: `The policy/head is trained and selected only on UCY train/val rows, but the underlying neural proposal, Stage37 floor, and teacher-repaired switch features are inherited from mixed external training. A full strict pure-UCY neural world-model retrain is still not complete.`

## Metrics

| split | rows | all | t50 | t100 | hard/failure | easy degradation | switch | collision delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `val` | 16103 | 0.2157 | 0.1963 | 0.2183 | 0.2176 | 0.0000 | 0.6152 | 0.0084 |
| `test` | 35441 | 0.2078 | 0.1674 | 0.2200 | 0.2098 | 0.0000 | 0.5038 | 0.0073 |

## Frozen Mixed-Policy UCY Test Reference

- metrics: `{'rows': 35441, 'all_improvement': 0.20210507661300892, 't10_improvement': 0.4683141081609017, 't25_improvement': 0.09290721224935328, 't50_improvement': 0.12383620122509942, 't100_improvement': 0.1495614315978797, 'hard_failure_improvement': 0.1973941742245735, 'easy_degradation': 0.0, 'harm_over_fallback': -0.10649609158273521, 'switch_rate': 0.3775570666741909, 'regret_to_oracle': -0.13220790863419538, 'by_domain': {'ETH_UCY': {'rows': 25901, 'all_improvement': 0.19062711907148822, 't50_improvement': 0.13578010021258757, 't100_improvement': 0.1345954848165677, 'hard_failure_improvement': 0.18707724155054317, 'easy_degradation': 0.0, 'switch_rate': 0.39380718891162503}, 'UCY': {'rows': 9540, 'all_improvement': 0.23265884628641076, 't50_improvement': 0.0913767291690647, 't100_improvement': 0.19255939765315455, 'hard_failure_improvement': 0.22601578755697505, 'easy_degradation': 0.0, 'switch_rate': 0.33343815513626834}}, 'alpha_mean': 0.31782624643774166, 'alpha_positive_rate': 0.3775570666741909, 'collision_delta_vs_floor_005': -0.002442598925256445, 'smoothness_jagged_delta': 0.0}`

## Source Inventory

`{'train': [{'source': 'UCY/students01/students001-trajnet.txt', 'rows': 47223, 't10': 15147, 't25': 13365, 't50': 11583, 't100': 7128, 'is_pure_ucy_source': True}, {'source': 'UCY/students03/obsmat.txt', 'rows': 70585, 't10': 19216, 't25': 18370, 't50': 17529, 't100': 15470, 'is_pure_ucy_source': True}], 'val': [{'source': 'UCY/zara01/obsmat.txt', 'rows': 16103, 't10': 4580, 't25': 4284, 't50': 3988, 't100': 3251, 'is_pure_ucy_source': True}], 'test': [{'source': 'UCY/zara02/obsmat.txt', 'rows': 25901, 't10': 7223, 't25': 6823, 't50': 6422, 't100': 5433, 'is_pure_ucy_source': True}, {'source': 'UCY/zara03/crowds_zara03.txt', 'rows': 9540, 't10': 3060, 't25': 2700, 't50': 2340, 't100': 1440, 'is_pure_ucy_source': True}]}`

## No Leakage

`{'future_endpoint_input': False, 'future_labels_eval_only': True, 'gain_harm_labels_train_only': True, 'validation_threshold_selection_only': True, 'test_threshold_tuning': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_endpoints_for_goal_construction': False, 'stage5c_executed': False, 'smc_enabled': False}`

Future endpoints are labels/evaluation only. The selected policy was chosen on UCY validation rows and evaluated once on UCY test rows.
