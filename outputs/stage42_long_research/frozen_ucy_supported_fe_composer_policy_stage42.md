# Frozen Stage42-FH UCY-Supported FE Composer Policy

- source: `fresh_stage42_fh_policy_freeze_replay`
- policy name: `stage42_fh_ucy_supported_fe_composer`
- policy hash: `f1f6e0636167fae8721a3f7195f188dcbe1a83194b04fa0625b378ad38b5aed6`
- selected candidate: `{'mode': 'fc_to_safety', 'fallback': 'di', 'scope': 'row', 'threshold': 0.05, 'margin': 0.01}`
- internal_val_group: `UCY::UCY/zara03/crowds_zara03.txt`
- frozen from: `outputs/stage42_long_research/ucy_supported_fe_composer_stage42.json`

## Runtime Inputs

- `predicted_fc_rollout_geometry`
- `predicted_di_fallback_geometry`
- `predicted_fa_fallback_geometry`
- `predicted_fb_fallback_geometry`
- `source_frame_horizon_group_key`
- `agent_id`
- `normalizer`

## Selection Discipline

- validation selection rule: `UCY train-only internal validation plus existing validation; candidate selected on validation only`
- test usage rule: `test labels used only for final evaluation and bootstrap, not for policy selection`
- no future endpoint input, no central velocity, no test endpoint goals.
- Stage5C false; SMC false; no metric/seconds claim.
