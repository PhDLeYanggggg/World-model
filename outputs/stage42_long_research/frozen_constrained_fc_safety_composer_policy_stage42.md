# Frozen Stage42-FE Constrained FC/Safety Composer Policy

- source: `fresh_stage42_fe_policy_freeze_replay`
- policy name: `stage42_fe_constrained_fc_safety_composer`
- policy hash: `a78db26aa155b38799f5b866f32a2d205018adf2054d9409a016da3163328dff`
- selected candidate: `{'mode': 'fc_to_safety', 'fallback': 'di', 'scope': 'row', 'threshold': 0.05, 'margin': 0.0025}`
- frozen from: `outputs/stage42_long_research/constrained_fc_safety_composer_stage42.json`

## Runtime Inputs

- `predicted_fc_rollout_geometry`
- `predicted_di_fallback_geometry`
- `predicted_fa_fallback_geometry`
- `predicted_fb_fallback_geometry`
- `source_frame_horizon_group_key`
- `agent_id`
- `normalizer`

## Selection Discipline

- validation selection rule: `candidate selected on validation only; test evaluated once`
- test usage rule: `test labels used only for final evaluation and bootstrap, not for policy selection`
- no future endpoint input, no central velocity, no test endpoint goals.
- Stage5C false; SMC false; no metric/seconds claim.
