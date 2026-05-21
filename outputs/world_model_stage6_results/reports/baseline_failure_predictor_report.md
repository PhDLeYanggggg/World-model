# Stage 6 Baseline Failure Predictor

| split | samples | positive_rate | AUROC | AUPRC | precision_at_k | recall_at_k | calibration_ECE | Brier | easy_false_alarm_rate | hard_recall | failure_type_F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 273 | 0.190476 | 0.951532 | 0.782588 | 0.730769 | 0.730769 | 0.06553 | 0.074779 | 0.104167 | 0.8 | not_available_no_semantic_ground_truth |
| val | 76 | 0.197368 | 0.935519 | 0.816668 | 0.8 | 0.8 | 0.088689 | 0.068479 | 0.114754 | 0.75 | not_available_no_semantic_ground_truth |
| test | 76 | 0.302632 | 0.899098 | 0.694048 | 0.73913 | 0.73913 | 0.154799 | 0.140132 | 0.064516 | 0.681818 | not_available_no_semantic_ground_truth |

