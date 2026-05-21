# Stage 7 Failure Predictor Comparison

| variant | samples | AUROC | AUPRC | calibration_ECE | Brier | hard_recall | easy_false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stage6_without_goal_scene_reference | 76 | 0.899098 | 0.694048 | 0.154799 | 0.140132 | 0.681818 | 0.064516 |
| without_goal_scene | 76 | 0.938474 | 0.780111 | 0.090185 | 0.104497 | 0.772727 | 0.193548 |
| with_goal_only | 76 | 0.886792 | 0.709867 | 0.179576 | 0.171727 | 0.863636 | 0.451613 |
| with_scene_only | 76 | 0.943396 | 0.81326 | 0.096625 | 0.097367 | 0.772727 | 0.258065 |
| with_goal_scene | 76 | 0.886792 | 0.709867 | 0.179576 | 0.171727 | 0.863636 | 0.451613 |
| with_goal_scene_interaction | 76 | 0.883511 | 0.683255 | 0.20488 | 0.183667 | 0.863636 | 0.451613 |
