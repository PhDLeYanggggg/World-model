# M3W Ablation Table

These are inference-time token masking ablations on the current hybrid checkpoint, not full retraining ablations.

| ablation | disabled tokens | t+50 improvement | hard/failure improvement | easy degradation |
| --- | --- | ---: | ---: | ---: |
| no_scene | scene_patch, scene_sdf | 0.016813 | 0.041009 | 0.012291 |
| no_goal | goal_region | 0.013409 | 0.023123 | 0.017494 |
| no_interaction | interaction_edge | 0.020152 | 0.044396 | 0.005810 |
| no_baseline_rollout | baseline_rollout | 0.084302 | 0.033995 | 0.016007 |

A full CCF-A candidate still requires retrained ablations and statistical tests.
