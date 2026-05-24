# M3W State Machine Stage A Freeze Manifest

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space benchmark，不是 metric benchmark。
- t+50 / t+100 是 raw annotation-frame horizon，不能说成 seconds-level。
- homography / scale / effective seconds 未验证。
- Stage5C 未执行。
- SMC 未启用。

- frozen model: `M3W-LAS v2 candidate`
- selected policy sha256: `792f0682ca2c8c74c2b0a366e3f9a2022223b70ef04ef2d2a38d64999d333d57`
- feature schema sha256: `de0b63a7e8d656e0f7177da3038023065e6d6e06835c31e0de34c27d64fda16b`
- frozen test arrays sha256: `1bec88fadfad1a2fa19298c77de53714ad6fa054b00240ede6cc3b6c702a1ab5`
- t+50 improvement: `0.1686288243790961`
- hard/failure improvement: `0.1336398986813968`
- easy degradation: `0.01928694490688554`
- no leakage: `pass`
- Stage5C executed: `False`
- SMC enabled: `False`
