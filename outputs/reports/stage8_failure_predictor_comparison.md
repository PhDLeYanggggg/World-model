# Stage 8 Failure Predictor v2

| variant | feature_mode | samples | failure_rate | AUROC | AUPRC | Brier | ECE | hard_recall | easy_false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage8_without_scene_goal | no_scene_goal | 60 | 0.316667 | 0.896021 | 0.773021 | 0.127704 | 0.157978 | 0.842105 | 0.146341 |
| stage8_with_scene_only | scene_only | 60 | 0.316667 | 0.896021 | 0.773021 | 0.127704 | 0.157978 | 0.842105 | 0.146341 |
| stage8_with_goal_only | goal_only | 60 | 0.316667 | 0.879332 | 0.754598 | 0.139682 | 0.164425 | 0.842105 | 0.146341 |
| stage8_with_scene_goal | scene_goal | 60 | 0.316667 | 0.879332 | 0.754598 | 0.139682 | 0.164425 | 0.842105 | 0.146341 |
| stage8_with_scene_goal_multiagent | scene_goal_multiagent | 60 | 0.316667 | 0.879332 | 0.754598 | 0.139682 | 0.164425 | 0.842105 | 0.146341 |
