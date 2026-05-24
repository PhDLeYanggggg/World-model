# M3W Long-Term State Machine Summary

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space benchmark，不是 metric benchmark。
- t+50 / t+100 是 raw annotation-frame horizon，不能说成 seconds-level。
- homography / scale / effective seconds 未验证。
- Stage5C 未执行。
- SMC 未启用。

- current stage: `F_plan_generated`
- Stage5C executed: `False`
- SMC enabled: `False`

| stage | verdict | passed |
| --- | --- | --- |
| A | stage_a_pass_enter_stage_b | True |
| B | stage_b_pass_enter_stage_c | True |
| C | stage_c_pass_enter_stage_d | True |
| D | stage_d_pass_enter_stage_e_pixel_raw_frame_only | True |
| E | stage_e_pass_enter_stage_f_plan_only | True |
| F | plan_generated | True |
