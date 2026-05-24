# Stage30 A Freeze Recheck

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space benchmark，不是 metric benchmark。
- t+50 / t+100 是 raw-frame horizon，不能说成 seconds-level。
- homography / scale / effective seconds 未验证，除非 Stage30 raw audit 证明。
- Stage5C 未执行。
- SMC 未启用。

- source: `fresh_run`
- t+50: `0.1686288243790961`
- hard/failure: `0.1336398986813968`
- easy degradation: `0.01928694490688554`
- cross_scene: `0.1064302019155895`
- within_scene: `0.15785983604964404`
- policy hash: `792f0682ca2c8c74c2b0a366e3f9a2022223b70ef04ef2d2a38d64999d333d57`
- schema hash: `b91443bbbfc175bc75edc8e16697abc7187a3de5159900670141fa8d0b583661`
- no leakage pass: `True`
- freeze gate pass: `True`
- source labels: `{'metric_recomputation': 'fresh_run', 'selected_arrays': 'cached_verified', 'feature_schema': 'cached_verified', 'latent_metadata': 'cached_verified'}`

## Agent Type Breakdown
- Pedestrian: `0.11016732108610128`
- Biker: `0.20528328459658607`
- Skater: `0.20918500363598935`
- Cart: `0.09415699732781369`
- Car: `-0.017048804306335708`
- Bus: `0.13007872802821607`
- unknown: `0.0`
