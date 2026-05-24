# Stage35 External Curriculum Adaptation Report

- source: `fresh_run`
- bounded adaptations: `{'hard_failure_oversampling': 'fresh_run_via_class_weighted_detectors', 'external_only_hard_finetune': 'diagnostic_same_policy', 'sdd_to_external_curriculum': 'diagnostic_not_deployable_without SDD easy pass', 'external_to_sdd_anti_forgetting': 'fallback_to_sdd_strongest_for_sdd', 'per_horizon_selector': 'not_run: external test t50 small and t100 absent', 'per_scene_selector': 'not_run: held-out scenes too few', 'pedestrian_only_selector': 'fresh_run_external_all_pedestrian'}`
- best metrics after adaptation: `{'rows': 66303, 'all_improvement': 0.12131890857784355, 't10_improvement': 0.30620476404756647, 't25_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.1398494448930071, 'easy_degradation': 0.0004114683717719725, 'selector_regret': 0.4289430569144169, 'switch_rate': 0.04999773765892946, 'mean_confidence': 0.03720055893063545}`
- curriculum helped: `True`
