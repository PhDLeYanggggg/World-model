# Stage35 Final Report

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。
- SDD remains pixel raw-frame; external remains dataset-local / unverified weak metric diagnostic.
- Stage5C executed: `False`
- SMC enabled: `False`

## What Ran

- External data expansion, split v2, hard/easy/failure labels, selective transfer policy, selector v3, curriculum adaptation, cross-domain eval, capability audit, and gates were run in this stage.
- Result sources are marked as `fresh_run`, `cached_verified`, or `not_run`; SDD-to-external zero-shot is not rebranded as successful transfer.
- converted external files: `18`
- external split rows: `{'train': {'rows': 158942, 'scenes': 6, 'scene_ids': ['ETH_UCY_seq_eth', 'ETH_UCY_seq_hotel', 'ETH_UCY_students03', 'ETH_UCY_zara01', 'ETH_UCY_zara02', 'TrajNet_biwi'], 'agents': 435, 'horizon_counts': {'10': 46604, '25': 42907, '50': 38943, '100': 30488}, 'track_length_median': 38.0}, 'val': {'rows': 112746, 'scenes': 2, 'scene_ids': ['TrajNet_crowds', 'TrajNet_mot'], 'agents': 892, 'horizon_counts': {'10': 37683, '25': 32059, '50': 26756, '100': 16248}, 'track_length_median': 20.0}, 'test': {'rows': 66303, 'scenes': 3, 'scene_ids': ['UCY_crowds', 'UCY_students01', 'UCY_zara03'], 'agents': 892, 'horizon_counts': {'10': 21267, '25': 18765, '50': 16263, '100': 10008}, 'track_length_median': 20.0}}`
- hard/easy/failure labels: `{'train': {'rows': 158942, 'easy': 39736, 'hard': 131433, 'failure': 64528, 'oracle_headroom': 0.46283990144729614}, 'val': {'rows': 112746, 'easy': 35967, 'hard': 78283, 'failure': 39068, 'oracle_headroom': 0.5199509859085083}, 'test': {'rows': 66303, 'easy': 20798, 'hard': 45917, 'failure': 22891, 'oracle_headroom': 0.5287383794784546}}`

## Best External Selector

- best selector metrics: `{'rows': 66303, 'all_improvement': 0.12131890857784355, 't10_improvement': 0.30620476404756647, 't25_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.1398494448930071, 'easy_degradation': 0.0004114683717719725, 'selector_regret': 0.4289430569144169, 'switch_rate': 0.04999773765892946, 'mean_confidence': 0.03720055893063545}`
- deployable by Stage35 criteria: `False`
- Interpretation: all-test and hard/failure are positive with easy preserved, but t+50 remains `0.0`, so this is not a deployable cross-domain M3W candidate.

## Cross-Domain Matrix

- cross-domain directions: `['SDD_to_SDD', 'SDD_to_external', 'external_to_external', 'external_to_SDD', 'SDD_external_to_SDD', 'SDD_external_to_external', 'held_out_external_scenes', 'external_hard_only', 'external_easy_only', 'external_t50']`
- `SDD_to_external` is marked `not_run` for the expanded Stage35 schema because the previous SDD zero-shot path failed and is not compatible with the new external expansion.
- `SDD_external_to_external` is marked `not_run`; Stage35 did not train a true mixed SDD+external selector.

## Capability Audit

- capability audit: `{'source': 'fresh_run', 'external_positive_transfer': True, 'still_sdd_specific': True, 'data_expansion_solved_horizon_shortage': True, 'selective_policy_protects_easy': True, 'goal_interaction_contribution': 'weak_or_not_proven', 'latent_predictive_value': False, 'cross_dataset_candidate': False, 'current_blockers': ['external t50 transfer gate failed', 'not deployable because all/t50/hard/easy gates are not all satisfied', 'goal/interaction weak', 'latent no predictive lift']}`
- gates: `12 / 14`
- verdict: `stage35_external_selective_transfer_not_deployable`
