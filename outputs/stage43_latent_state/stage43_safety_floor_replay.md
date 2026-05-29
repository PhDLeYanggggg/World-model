# Stage43-A Safety Floor Freeze and Exact Replay

- source: `fresh_stage43_a_safety_floor_replay`
- result source: Stage42 current floor `fresh_run`, historical floors `cached_verified`
- verdict: `stage43_a_safety_floor_replay_pass`
- gate: `14 / 14`
- latent-state training precondition: `True`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local/raw-frame 2.5D 多智能体 world-state candidate。
- Stage43-A 只做训练前 safety-floor freeze/replay，不训练 latent-state model。
- Stage26 SDD、Stage37 external t50、M3W-Neural v1 和 Stage42 source/full-waypoint protected policy 均作为安全地板证据冻结。
- future endpoint / waypoint 只能作为 supervised/evaluation label，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoint 构建 goals，不使用 test metric 调 threshold。
- t+50/t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local / pixel-space 不能写成 metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Frozen Floors

- Stage26 SDD selector: cached_verified; remains SDD pixel-space floor.
- Stage37 external t50 selector: cached_verified; frozen policy hash recorded.
- M3W-Neural v1 protected composite: cached_verified; manifest/policy hash recorded.
- Stage42 source/domain full-waypoint protected policy: fresh row-cache replay in this run.

## Fresh Stage42 Replay

- rows: `47458`
- domains: `{'TrajNet': 37918, 'UCY': 9540}`
- horizon counts: `{'10': 15402, '25': 13470, '50': 11538, '100': 7048}`
- ADE all improvement: `0.291543`
- ADE t+50 improvement: `0.247045`
- ADE t+100 raw-frame diagnostic improvement: `0.196335`
- ADE hard/failure improvement: `0.287273`
- easy degradation: `0.000000`
- fallback exact floor rate: `1.000000`
- row hash: `63459ece3d6578cfbce35288fc6a1f95193e4e20252a33e319ef4478d760b1c7`
- max replay diff vs Stage42-IV report: `0.000000060023`

## Gate

| gate | passed |
| --- | --- |
| stage26_floor_evidence_frozen | True |
| stage37_policy_frozen | True |
| m3w_neural_v1_manifest_frozen | True |
| stage42_cache_exists | True |
| stage42_exact_replay_diff_zero | True |
| stage42_all_t50_t100_hard_positive | True |
| easy_preserved | True |
| fallback_exact_floor_rate_ok | True |
| source_domains_present | True |
| row_hash_recorded | True |
| no_future_or_test_leakage | True |
| no_metric_seconds_3d_or_foundation_claim | True |
| stage5c_false | True |
| smc_false | True |

## Decision

Safety floor replay is closed; Stage43 latent-state dataset construction may start next under this frozen floor.

Stage43-A does not execute Stage5C, does not enable SMC, and does not claim metric/seconds/true-3D/foundation status.
