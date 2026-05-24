# M3W State Machine Stage D Time/Geometry Audit

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space benchmark，不是 metric benchmark。
- t+50 / t+100 是 raw annotation-frame horizon，不能说成 seconds-level。
- homography / scale / effective seconds 未验证。
- Stage5C 未执行。
- SMC 未启用。

- conclusion: `pixel-space only, raw-frame horizon only`
- metric claim allowed: `False`
- seconds-level claim allowed: `False`
