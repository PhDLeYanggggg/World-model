# Stage42-JS Source Context Gain/Harm Closure

- source: `fresh_stage42_js_source_context_gain_harm_closure`
- generated_at_utc: `2026-05-29T04:46:07.320545+00:00`
- git_commit: `0ca07d9`
- input_hash: `a41a36a011d25eaff280f4c00c500eb953d2e1edf42de74120dd399aa6e12b0a`
- gate: `14 / 14`
- verdict: `stage42_js_source_context_gain_harm_closure_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JS consolidates fresh JR/IO/IP/IQ/IR evidence after rerunning the gain/harm repair chain on current HEAD.
- This closes the current source-level sequence/graph context candidate family as an independent t50/t100 main claim.
- h10/h25 have narrow horizon-specific positive routing evidence, but t50/t100 remain unsupported under this candidate family.
- future endpoints / waypoints are labels/evaluation only, not inference input.
- No central velocity, no test endpoint goals, no test threshold tuning.
- No metric/seconds claim, no Stage5C execution, no SMC.

## Summary

- decision: `close_current_source_sequence_graph_gain_harm_family_for_t50_t100_main_claim`
- deployment_decision: `keep_baseline_family_and_existing_protected_floor_as_deployable_context_mechanism`
- narrow_positive_horizon_routers: `['h10_history_only', 'h10_motion_goal_context', 'h25_baseline_plus_history_goal_neighbor']`
- t50_diagnosis: `router_under_switches_despite_headroom`; oracle_headroom `0.035246`
- t100_diagnosis: `weak_predictive_signal_or_baseline_family_dominance`; oracle_headroom `0.011163`
- IQ repair supported: `False`; best `baseline_plus_history_goal_neighbor__gain_only`; t50 `0.000001`
- IR repair supported: `False`; best `history_only__gain_only`; t50 `0.000000`
- next_repair_direction: `new candidate policies or row-level/source-slice objectives; not more threshold tuning on the same sequence/graph proposals`

## Horizon Router Summary

| horizon | best | candidate | all | t50 | t100 raw | hard/failure | easy | switch | supported |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | `h10_motion_goal_context` | `motion_goal_context` | 0.069270 | 0.000000 | 0.000000 | 0.072655 | -0.035269 | 0.444293 | `True` |
| 25 | `h25_baseline_plus_history_goal_neighbor` | `baseline_plus_history_goal_neighbor` | 0.006986 | 0.000000 | 0.000000 | 0.016655 | -0.021896 | 0.104751 | `True` |
| 50 | `h50_baseline_plus_history_goal_neighbor` | `baseline_plus_history_goal_neighbor` | 0.000001 | 0.000001 | 0.000000 | 0.000001 | -0.000000 | 0.015947 | `False` |
| 100 | `h100_history_only` | `history_only` | 0.001448 | 0.000000 | 0.001448 | 0.001448 | 0.011562 | 0.128405 | `False` |

## Failure Taxonomy

- residual_protocol: `JR confirms sequence/graph residual variants degrade all/t50/hard versus baseline-family rollout context.`
- horizon_mixing: `IO shows h10/h25 narrow positives, but t50/t100 remain unsupported; horizon mixing is only a partial explanation.`
- t50_blocker: `router_under_switches_despite_headroom`
- t100_blocker: `weak_predictive_signal_or_baseline_family_dominance`
- gain_harm_repair: `IQ validation-selected gain/harm calibration still fails to capture t50 headroom.`
- source_pattern_repair: `IR source-pattern support also falls back to no useful t50 switches.`
- what_not_to_claim: `['source-level sequence/graph context is an independent t50/t100 contribution', 'scene/goal/neighbor/interaction has been proven as a main driver by this candidate family', 'more threshold tuning on the same candidate proposals is likely enough']`

## Gate

| gate | pass |
| --- | ---: |
| `jr_negative_residual_replay_loaded` | `True` |
| `io_horizon_router_loaded` | `True` |
| `ip_blocker_audit_loaded` | `True` |
| `iq_gain_harm_repair_evaluated` | `True` |
| `ir_source_pattern_repair_evaluated` | `True` |
| `narrow_h10_h25_positive_recorded` | `True` |
| `t50_repair_not_supported_recorded` | `True` |
| `t100_blocker_recorded` | `True` |
| `current_candidate_family_closed_for_t50_t100` | `True` |
| `next_action_not_more_threshold_tuning` | `True` |
| `negative_result_not_overclaimed` | `True` |
| `no_metric_seconds_overclaim` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |
