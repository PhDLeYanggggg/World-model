# Stage31 Final Report

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。
- SDD 是 pixel raw-frame benchmark，不是 metric seconds benchmark。
- Stage5C 未执行。
- SMC 未启用。

- source labels are recorded in JSON reports; raw external files and checkpoints are cached_verified inputs, conversion/eval/adaptation are fresh_run.
- zero-shot M3W-LAS external all improvement: `-0.9266750268846149`
- zero-shot M3W-LAS external t50 improvement: `-2.78572783999095`
- adapted M3W-LAS external all improvement: `0.0`
- adapted M3W-LAS external t50 improvement: `0.0`
- domain gap status: `SDD_candidate_with_external_adapted_diagnostic`
- gates: `10 / 11`
- tests: `python -m pytest tests` -> `56 passed`
- verdict: `stage31_external_domain_gap_sdd_candidate_only`
- External coordinates remain dataset-local; no metric/seconds claim.
