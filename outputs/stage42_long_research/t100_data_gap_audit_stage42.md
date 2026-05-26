# Stage42-BB T100 Data Gap Audit

- source: `fresh_synthesis_from_stage42_ba_and_calibration`
- generated_at_utc: `2026-05-26T11:55:23.301162+00:00`
- git_commit: `de6d60e`
- input_hash: `a69ff8ba72d4c2843ccc9bb2d76e6747d1bb2169a94bc19a1ef058faa5e31794`
- gate: `14 / 14`
- verdict: `stage42_bb_t100_data_gap_audit_pass_with_data_blocker`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BB 是 Stage42-BA 后的 t100 数据/标定缺口审计，不重新训练模型。
- t100 positive gain 在 Stage42-BA train-only source-CV 下缺少独立 source 支持。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t100 仍是 raw-frame diagnostic，不是 seconds-level long-horizon claim。
- metric / seconds-level pedestrian claims 仍被禁止，除非未来完成官方 FPS/stride/homography/scale 验证。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Stage42-BA Dependency

- ba_verdict: `stage42_ba_t100_source_cv_repair_pass_with_t100_blocker`
- after_source_cv_guard_all: `0.280997`
- after_source_cv_guard_t50: `0.289698`
- after_source_cv_guard_t100_raw_frame_diagnostic: `0.000000`
- after_source_cv_guard_hard_failure: `0.251576`
- after_source_cv_guard_easy_degradation: `-0.372431`

## T100 Source Support Gaps

| domain | folds | safe-positive folds | supported | extra sources needed | blocker |
| --- | ---: | ---: | ---: | ---: | --- |
| `ETH_UCY` | 4 | 0 | `False` | 2 | t100_easy_safety_not_stable_across_source_cv |
| `TrajNet` | 3 | 1 | `False` | 1 | t100_easy_safety_not_stable_across_source_cv |
| `UCY` | 0 | 0 | `False` | 1 | insufficient_t100_capable_original_train_sources |

## Dataset Actions

| dataset | raw | converted | calibration | metric | seconds | next action |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `sdd` | `True` | `True` | pixel_raw_frame_only | `False` | `False` | user_action_or_source_specific_repair_required |
| `opentraj` | `True` | `True` | calibration_files_found_but_not_validated | `False` | `False` | user_action_or_source_specific_repair_required |
| `eth_ucy` | `True` | `True` | calibration_files_found_but_not_validated | `False` | `False` | user_action_or_source_specific_repair_required |
| `trajnet` | `True` | `True` | not_verified | `False` | `False` | user_action_or_source_specific_repair_required |
| `ucy` | `True` | `True` | calibration_files_found_but_not_validated | `False` | `False` | user_action_or_source_specific_repair_required |
| `tgsim` | `True` | `True` | traffic_metric_diagnostic_only | `True` | `False` | user_action_or_source_specific_repair_required |
| `aerialmpt` | `False` | `True` | not_verified | `False` | `False` | user_action_or_source_specific_repair_required |

## Summary

- unsupported_t100_domains: `['ETH_UCY', 'TrajNet', 'UCY']`
- supported_t100_domains: `[]`
- additional_t100_sources_needed_by_domain: `{'ETH_UCY': 2, 'TrajNet': 1, 'UCY': 1}`
- final_all_positive_after_guard: `True`
- final_t50_positive_after_guard: `True`
- final_hard_positive_after_guard: `True`
- final_easy_safe_after_guard: `True`
- final_t100_positive_after_guard: `False`
- global_metric_claim_allowed: `False`
- global_seconds_claim_allowed: `False`

## Interpretation

- Stage42-BB does not train a new model. It turns the Stage42-BA t100 source-CV blocker into an actionable data/calibration gap report.
- Protected all/t50/hard remain positive after the source-CV guard, but t100 positive gain is not supported by enough independent train-only source-CV evidence.
- The correct deployment/paper posture is to keep t100 as raw-frame diagnostic blocker until additional independent t100-capable sources or source-specific support are available.
- Metric and seconds-level claims remain rejected for pedestrian/top-down domains until FPS/stride/homography/scale are verified from official sources.
