# User / Engineering Action Required: T100 Runtime Batch Replay Row Cache

Stage42-HU found that Stage42-HR/HS/HT artifacts are sufficient for frozen policy replay and runtime smoke tests, but not for real row-level batch replay.

Required cache fields:

- `row_id`
- `split`
- `domain`
- `source_file`
- `scene_id_optional`
- `frame_id`
- `agent_id`
- `horizon`
- `candidate_xy_predicted_rollout`
- `floor_xy_train_horizon_causal_rollout`
- `original_selected_xy_from_stage42_hr_optional`
- `candidate_switch_optional`
- `future_xy_label_eval_only_optional`
- `normalizer_optional`
- `hard_failure_label_optional`
- `easy_label_optional`

Rules:

- `future_xy_label_eval_only_optional` may be used only for loss/evaluation, never as runtime input.
- `candidate_xy_predicted_rollout` and `floor_xy_train_horizon_causal_rollout` must be stored with stable row ids.
- Thresholds must remain validation-selected; test rows are replay/evaluation only.
- This remains raw-frame/dataset-local evidence unless metric/time calibration is separately verified.
