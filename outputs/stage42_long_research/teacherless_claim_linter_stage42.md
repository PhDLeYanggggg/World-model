# Stage42-HG Teacherless / Floor-Free Claim Linter

- source: `fresh_stage42_hg_teacherless_claim_linter`
- generated_at_utc: `2026-05-27T17:14:22.361597+00:00`
- git_commit: `b1cf5df`
- input_hash: `b971e9aac8ed3ac0f4d57d8648e608f808fb61b7b13b3d7fbd3a97045e4a2487`
- gate: `15 / 15`
- verdict: `stage42_hg_teacherless_claim_linter_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-HG 是 teacherless/floor-free claim linter，不训练、不转换、不调 threshold。
- 允许表述：teacherless proximity-guarded switch gate with causal floor fallback。
- 禁止表述：global floor-free neural deployment、causal floor removal、ungated neural deployment。
- future endpoint / future waypoint 只可作为 supervised/evaluation labels，不可作为 inference input。
- 不使用 central velocity，不使用 test endpoints 建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- files_scanned: `18`
- violations_total: `0`
- allowed_phrase_hits: `17`
- causal_floor_fallback_phrase_hits: `39`
- global_floor_free_claim_allowed: `False`
- ungated_neural_deployment_allowed: `False`

## Violations By Check

| check | count |
| --- | ---: |
| `teacherless_as_global_floor_free` | 0 |
| `floor_free_deployable_overclaim` | 0 |
| `ungated_neural_deployable` | 0 |
| `causal_floor_removal_allowed` | 0 |
| `metric_seconds_true3d_foundation` | 0 |
| `stage5c_smc_enabled` | 0 |

## Violations

- None.

## Gate

| gate | pass |
| --- | ---: |
| `source_fresh` | True |
| `hf_contract_loaded` | True |
| `he_input_passed` | True |
| `hc_input_passed` | True |
| `files_scanned` | True |
| `no_teacherless_global_floor_free_overclaim` | True |
| `no_floor_free_deployable_overclaim` | True |
| `no_ungated_neural_deployable_overclaim` | True |
| `no_causal_floor_removal_allowed_overclaim` | True |
| `no_metric_seconds_true3d_foundation_overclaim` | True |
| `no_stage5c_smc_overclaim` | True |
| `allowed_phrase_present` | True |
| `required_floor_phrase_present` | True |
| `stage5c_false` | True |
| `smc_false` | True |
