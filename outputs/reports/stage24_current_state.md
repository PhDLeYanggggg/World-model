# Stage 24 Current State

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D per-agent multi-agent trajectory world-state scaffold。
- SDD 是 pixel-space official benchmark，不是 metric benchmark。
- t+50 / t+100 是 raw annotation-frame horizon；effective seconds unknown。
- Stage 23 是 quick-plus，不能替代 medium/full。
- Stage 5C latent generative 仍禁止；SMC 仍禁止。

- 为什么 Stage 23 quick-plus 不能替代 medium：`quick-plus evaluated a reduced baseline cap and is explicitly partial; it cannot replace true medium statistics.`
- 为什么下一步先修 I/O：`Stage23 bottleneck was repeated compressed NPZ random access for per-agent start/target frames; medium requires fast per-video cache and frame/agent indexes.`
- 当前 SDD 数据量：`{'scenes': 8, 'videos': 60, 'tracks': 10300, 'rows': 10616256}`
- 当前 strongest baseline：`damped_velocity on Stage22/Stage23 quick-plus cross-scene; within_scene quick-plus also exposed scene_clamped for longer horizons`
- selector/failure/JEPA 状态：`{'selector_t50_improvement': 0.02656760613375428, 'failure_auroc': 0.64975, 'jepa_downstream_lift': 0.0}`
