# Stage 17 User Action Required

当前 selector oracle 有理论空间，但 trained selector/correction specialist 未过 gate。继续模型训练收益很低，优先补数据/标注。

需要用户提供：
- Provide Stanford Drone Dataset local path after accepting its license.
- Provide OpenTraj/full pedestrian-drone local path if available.
- Human-confirm high-value Stage16 annotation tasks into human silver/gold labels.

原因：
- Current data lets oracle baseline selection improve, but learned causal selector does not generalize enough.
- Strongest baseline explains most easy trajectories; hard/failure rows remain limited.
- Scene/goal/interaction labels are not strong enough to support reliable correction.

目标数量：
- official_t100_rows: 200+
- hard_failure_rows: 100+
- human_confirmed_scenes: at least 3, preferably 10+
