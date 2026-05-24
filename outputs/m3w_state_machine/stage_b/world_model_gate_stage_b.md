# M3W State Machine Stage B Gates

- gates passed: `5 / 5`
- current verdict: `stage_b_pass_enter_stage_c`
- Stage5C execution: `False`
- SMC enabled: `False`

| gate | pass | evidence |
| --- | --- | --- |
| B1 CI t+50 Above Stage26 | True | 0.16006103130418847 > 0.14583655843823773 |
| B2 CI Hard/Failure Above Stage26 | True | 0.12780886662468002 > 0.11234058960663984 |
| B3 Cross-Scene Does Not Collapse | True | cross_scene CI low 0.09883479819529153 > 0 |
| B4 Easy Preservation | True | 0.01928694490688554 <= 0.02 |
| B5 External Validation Or Clear Blocker | True | No converted non-SDD top-down feature store aligned to M3W-LAS exists yet; do not fabricate external validation. |
