# Stage42-CI Context Contribution Forensics

- source: `fresh_synthesis_from_stage42_ablation_and_claim_audits`
- generated_at_utc: `2026-05-26T20:35:04.100508+00:00`
- git_commit: `16fed1c`
- input_hash: `c38d39b241211311b6876fd5bc7afd0cb88d14a9d11903d0698b6588ac46b127`
- gate: `13 / 13`
- verdict: `stage42_ci_context_contribution_forensics_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CI 是 context contribution forensics，不下载、不转换、不执行 Stage5C、不启用 SMC。
- 本审计整合 retrained ablation / residual / neural / sequence / graph context 证据，防止把 mixed context 结果包装成主贡献。
- future endpoints / waypoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。

## Context Rows

| module | status | main claim? | evidence | next action |
| --- | --- | ---: | --- | --- |
| `baseline_family_rollout_context` | `supported_dominant_mechanism` | `True` | AU verdict=baseline_family_rollout_context_supported_as_dominant_mechanism; baseline_family_only all=0.287773; t50=0.315425; hard=0.275812 | Keep as the current dominant mechanism and use it as the teacher/floor for any stronger neural dynamics experiment. |
| `history_tokens` | `supported_core_component` | `True` | sequence full-minus-no-history t50=0.457817; hard=0.470799; matrix status=positive_contribution | Keep sequence history tokens in paper method and future models; do not reduce them to only flattened residual features. |
| `domain_expert` | `supported_secondary_component` | `True` | sequence full-minus-no-domain t50=0.041885; hard=0.039867; matrix status=positive_contribution | Use domain expert as a guarded source/horizon conditioning module, not as a broad generalization claim. |
| `goal_scene_context` | `mixed_partial_not_main_claim` | `False` | AO positive standalone=['history_only', 'motion_goal_context']; AO positive incremental=[]; sequence full-minus-no-goal t50=-0.004259; Z C5 status=mixed_not_main_claim | Try a source/horizon validation-gated goal prototype expert rather than global goal/scene injection; require bootstrap-positive t50 and easy<=2 before main claim. |
| `neighbor_interaction_context` | `mixed_weak_not_main_claim` | `False` | AS verdict=stage42_as_graph_context_not_supported; sequence full-minus-no-neighbor all=-0.000078; hard=-0.001343; matrix status=positive_contribution | Only keep neighbor features as auxiliary diagnostics unless a stronger graph-neural trial beats baseline-family with easy preservation. |
| `jepa_auxiliary` | `negative_or_inconclusive` | `False` | matrix no_JEPA status=cached_negative_or_inconclusive; interpretation=JEPA-only architecture evidence is cached verified and negative/unsafe; Stage42 does not have a fresh no-JEPA retrain that supports a positive JEPA claim. | Keep JEPA as diagnostic/auxiliary only until a downstream probe improves protected metrics. |
| `transformer_dynamics` | `negative_or_inconclusive_as_independent_claim` | `False` | matrix no_Transformer status=negative_or_inconclusive; delta_t50=-0.006705 | Future Transformer work should target full-waypoint source-level lift under Stage37 floor, then test floor relaxation separately. |
| `stage37_teacher_floor_and_safe_switch` | `supported_necessary_safety_mechanism` | `True` | ungated easy degradation=1.245861; no_teacher_floor status=negative_unsafe; no_safe_switch status=positive_contribution | Frame the floor as a safety mechanism and current core contribution; only relax it on validation-supported source/horizon slices. |

## Failure Taxonomy

- Baseline-family rollout context is currently the dominant supported mechanism, not merely a nuisance fallback.
- History tokens are supported when encoded as causal sequence context.
- Domain expert is supported as a smaller guarded source/horizon component.
- Goal/scene context has standalone signal but is not reliably incremental after baseline-family context.
- Neighbor/interaction context is weak/mixed under current hand-built graph and sequence protocols.
- JEPA and Transformer evidence remains diagnostic/protected, not independent floor-free deployment evidence.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'labels_eval_only': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
