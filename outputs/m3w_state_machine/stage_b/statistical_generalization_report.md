# M3W State Machine Stage B Statistical And Generalization Report

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space benchmark，不是 metric benchmark。
- t+50 / t+100 是 raw annotation-frame horizon，不能说成 seconds-level。
- homography / scale / effective seconds 未验证。
- Stage5C 未执行。
- SMC 未启用。

- bootstrap samples: `2000`
- t+50 CI: `{'mean': 0.16865755284543865, 'ci_low': 0.16006103130418847, 'ci_high': 0.1774508372034304, 'n': 24810}`
- hard/failure CI: `{'mean': 0.1336479491598302, 'ci_low': 0.12780886662468002, 'ci_high': 0.1387674747628355, 'n': 96581}`
- cross_scene CI: `{'mean': 0.10638971037780987, 'ci_low': 0.09883479819529153, 'ci_high': 0.11367006756786821, 'n': 50000}`
- within_scene CI: `{'mean': 0.1579360667904667, 'ci_low': 0.15036045747067164, 'ci_high': 0.16506592523478197, 'n': 50000}`
- easy degradation point: `0.01928694490688554`
- external validation completed: `False`
- external blocker: No converted non-SDD top-down feature store aligned to M3W-LAS exists yet; do not fabricate external validation.

## Per-Agent-Type Improvement
- Pedestrian: `0.11016732108610128`
- Biker: `0.20528328459658607`
- Skater: `0.20918500363598935`
- Cart: `0.09415699732781369`
- Car: `-0.017048804306335708`
- Bus: `0.13007872802821607`
- unknown: `0.0`
