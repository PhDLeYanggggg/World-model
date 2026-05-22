# Final Model Metrics

- final_selection: `strongest_baseline_fallback`
- official_horizon: `t+50`
- t+100 status: `diagnostic`
- official FDE@50 improvement: `0.000000`
- diagnostic FDE@100 improvement: `0.000000`
- hard/failure improvement: `0.000000`
- easy degradation: `0.000000`
- physical validity: `preserved`

| model | subset | horizon | improvement | official |
| --- | --- | ---: | ---: | --- |
| strongest_causal_baseline | all | 50 | 0.000000 | True |
| stage15_best | all | 50 | 0.005442 | True |
| stage16_best | all | 50 | 0.009176 | True |
| final_without_fallback | all | 50 | 0.009176 | True |
| final_with_fallback | all | 50 | 0.000000 | True |
| final_without_fallback | all | 100 | 0.011476 | False |
| final_with_fallback | all | 100 | 0.000000 | False |
| final_without_fallback | hard | 50 | 0.011476 | True |
| final_without_fallback | baseline_failure | 50 | 0.007894 | True |
| no_scene_ablation | hard | 50 | 0.011476 | True |
| no_goal_ablation | hard | 50 | 0.011476 | True |
| no_interaction_ablation | hard | 50 | 0.011476 | True |
