# Stage42-CU Runtime Policy API Smoke Audit

- source: `fresh_runtime_api_from_frozen_policy_artifact`
- generated_at_utc: `2026-05-26T19:46:57.462514+00:00`
- git_commit: `728ad2b`
- policy_hash: `4af6536f86499d5b39efa535bb81978398586d65746bb983571b642af7c92d59`
- gate: `19 / 19`
- verdict: `stage42_cu_runtime_policy_api_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CU 把 Stage42-CS frozen policy artifact 变成可调用 runtime policy API。
- runtime policy 只使用 domain、horizon 和模型预测 rollout geometry 的 group min-distance。
- runtime policy 不使用 future endpoint、future waypoints、central velocity 或 test endpoint goals。
- runtime smoke audit 是 deployment/reproducibility evidence，不是新增模型训练分数。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Runtime API

- loader: `FrozenProximityGuardPolicy.from_file()`
- decision method: `policy.decide(domain=..., horizon=..., endpoint_min_group_distance=..., full_min_group_distance=...)`
- inputs: `domain`, `horizon`, predicted endpoint group min-distance, predicted full-waypoint group min-distance.
- output: deterministic `RuntimeDecision` with `use_full_waypoint`, `guarded_off`, and reason.

## Smoke Cases

| case | decision | reason | passed |
| --- | --- | --- | ---: |
| `full_slice_guard_clear` | use_full=`True`, guarded_off=`False` | `base_choice_full_waypoint_guard_clear` | `True` |
| `full_slice_guarded_off` | use_full=`False`, guarded_off=`True` | `proximity_guard_fallback_to_endpoint_linear` | `True` |
| `endpoint_slice_never_switches` | use_full=`False`, guarded_off=`False` | `base_choice_endpoint_linear` | `True` |
| `full_slice_nonfinite_geometry_replays_no_guard` | use_full=`True`, guarded_off=`False` | `base_choice_full_waypoint_geometry_nonfinite_replay_no_guard` | `True` |

## Interpretation

- Stage42-CU turns the frozen policy artifact into a callable deployment component.
- It does not reselect thresholds and does not add new model scores.
- Nonfinite predicted geometry follows the exact CQ replay behavior: if the base slice selected full-waypoint, the guard does not fire.
- This remains protected dataset-local/raw-frame 2.5D evidence, not true 3D, not metric/seconds-level, not Stage5C, and not SMC.
