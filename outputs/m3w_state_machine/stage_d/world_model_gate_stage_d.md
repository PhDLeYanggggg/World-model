# M3W State Machine Stage D Gates

- gates passed: `3 / 3`
- current verdict: `stage_d_pass_enter_stage_e_pixel_raw_frame_only`
- Stage5C execution: `False`
- SMC enabled: `False`

| gate | pass | evidence |
| --- | --- | --- |
| D1 Effective Seconds Audit | True | Evidence insufficient; reports keep raw-frame only. |
| D2 Metric/Homography Audit | True | Evidence insufficient; reports keep pixel-space only. |
| D3 Report Claim Safety | True | All state-machine reports explicitly forbid metric/seconds/true-3D claims. |
