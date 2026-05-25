# Stage42-D Causal Ablation Evidence Audit

- source: `fresh_run`
- generated_at_utc: `2026-05-25T20:08:00.255378+00:00`
- git_commit: `c26d980`
- input_hash: `4ae781add304d5797c95cae26525f32b77a985bd89d23f6199ca36b84c863bc4`

## Claim Boundary

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- SDD 仍是 pixel-space；external 仍是 dataset-local / unverified weak metric diagnostic。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- future endpoints / future waypoints 只作为 loss/eval label，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Fresh Stage42 Ablation Rows

| ablation | source | status | all | t50 | t100 diag | hard/failure | easy degr | switch | delta all | delta t50 | interpretation |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `no_neural_tail_use_teacher_floor_only` | `fresh_run` | `positive_safe` | 0.2036 | 0.1312 | 0.1337 | 0.1966 | -0.1445 | 0.2954 | -0.0067 | -0.0054 | Removing the composite-tail neural safe switch leaves the Stage37/teacher floor; positive protected-minus-teacher deltas indicate neural tail contribution. |
| `no_safe_floor_use_ungated_endpoint_neural` | `fresh_run` | `negative_unsafe` | 0.2966 | 0.2152 | 0.3592 | 0.3294 | 1.2459 | 1.0000 | 0.0864 | 0.0787 | Ungated endpoint neural is a no-fallback safety ablation; it can improve raw all but is not deployable if easy degradation exceeds 2%. |
| `oracle_floor_vs_neural_diagnostic` | `fresh_run` | `positive_safe` | 0.4222 | 0.3452 | 0.4260 | 0.4211 | -0.2999 | 0.5740 | 0.2120 | 0.2087 | Diagnostic oracle uses future labels only to measure remaining headroom; it is not a deployable model. |
| `no_full_waypoint_sequence_use_endpoint_linear_bridge` | `fresh_run` | `positive_safe` | 0.2103 | 0.1365 | 0.1469 | 0.2038 | -0.1451 | 0.3410 | 0.0245 | -0.0115 | Endpoint-linear bridge removes the full-waypoint sequence model. delta_vs_reference is ablation-minus-protected: negative t50/t100 deltas mean the full-waypoint model helps those horizons, while positive all-delta means endpoint-linear remains stronger on all-ADE. |
| `no_safe_floor_use_ungated_full_waypoint` | `fresh_run` | `negative_unsafe` | 0.2966 | 0.2152 | 0.3592 | 0.3294 | 1.2459 | 1.0000 | 0.1108 | 0.0672 | Ungated full-waypoint neural is a no-fallback safety ablation; it remains diagnostic if easy degradation is unsafe. |
| `no_composite_tail_use_teacher_linear_bridge` | `fresh_run` | `positive_safe` | 0.2036 | 0.1312 | 0.1337 | 0.1966 | -0.1445 | 0.2954 | 0.0178 | -0.0169 | Teacher linear bridge is the pre-composite floor in waypoint space; protected full-waypoint must improve without easy harm. |

## Cached-Verified Required Ablation Coverage

| ablation | source | status | evidence type | evidence source | interpretation |
| --- | --- | --- | --- | --- | --- |
| `no_history` | `cached_verified` | `complete` | fresh_run masked-feature ablation | outputs/stage41_fresh_confirmation/stage41_teacher_guided_evidence.json | Masks history/static causal feature group after policy freeze; proves coverage for no-history/static ablation, not a claim that every history-derived scalar is useless. |
| `no_neighbor` | `cached_verified` | `complete` | fresh_run masked neighbor/interaction ablations | teacher_guided_evidence + group_consistency_evidence | Neighbor/interaction masking is audited; group/neighbor features are especially important for the safety head and t100/hard slices. |
| `no_scene_goal` | `cached_verified` | `complete` | fresh_run scene/goal proxy and route/physical ablations | stage41_teacher_guided_evidence + stage41_route_physical_policy_integration | Scene/goal proxy coverage exists. Current deployable trajectory path keeps route/physical mostly diagnostic; route/physical heads are not main trajectory deployment claims. |
| `no_interaction` | `cached_verified` | `complete` | fresh_run interaction/group-consistency masking | stage41_teacher_guided_evidence + stage41_group_consistency_evidence | Interaction/group-consistency features have explicit ablations and are necessary for guarded deployment; without them raw neural remains less safe. |
| `no_jepa` | `cached_verified` | `complete_with_same_protocol_negative_evidence` | fresh JEPA disable decision plus same-protocol Stage41 pure-Transformer negative attempts | stage41_jepa_deployment_decision + stage41_neural_architecture_ablation_audit + stage30_m3w_verified/retrained_ablation_fresh | JEPA is explicitly disabled from the deployable path because audited JEPA variants were non-collapse but did not give deployable downstream lift. Stage41 now also records same-protocol pure-Transformer/no-JEPA attempts as negative or fallback-only, so the current positive path is protected endpoint neural dynamics rather than JEPA/Transformer purity. |
| `no_transformer` | `cached_verified` | `complete_with_same_protocol_negative_evidence` | same-protocol JEPA-only negative attempts plus historical cross-protocol ablation | stage41_neural_architecture_ablation_audit + stage30_m3w_verified/retrained_ablation_fresh + Stage41 JEPA-only diagnostics | Stage41 same-protocol JEPA-only/no-Transformer attempts are negative or unsafe, so no-Transformer is covered as negative architecture evidence. This is not a claim that JEPA contributes to the deployable path; it is why JEPA remains diagnostic-only. |
| `no_fallback` | `cached_verified` | `complete` | fresh no-fallback negative safety ablation | teacher_guided_evidence + pure_ucy_neural_statistical_evidence + all_agent_composite_world_state | No-fallback neural often improves hard/all raw error but catastrophically damages easy cases; fallback is required for deployability. |

## Cached-Verified Architecture Ablation

| architecture | source | attempted | candidates | best | deployable | interpretation |
| --- | --- | ---: | ---: | --- | ---: | --- |
| `transformer_only` | `cached_verified` | `True` | 5 | `Stage41_conformal_safety_head_transformer` | `False` | same-protocol cached verified architecture audit; current positive path is protected endpoint/full-waypoint dynamics |
| `jepa_only` | `cached_verified` | `True` | 1 | `Stage41_jepa_auxiliary_representation` | `False` | same-protocol cached verified architecture audit; current positive path is protected endpoint/full-waypoint dynamics |
| `hybrid_jepa_transformer` | `cached_verified` | `True` | 3 | `Stage41_easy_guard_distilled_hybrid` | `False` | same-protocol cached verified architecture audit; current positive path is protected endpoint/full-waypoint dynamics |
| `mixture_selector` | `cached_verified` | `True` | 1 | `Stage41_mixture_of_experts_baseline_selector` | `False` | same-protocol cached verified architecture audit; current positive path is protected endpoint/full-waypoint dynamics |
| `protected_neural_endpoint` | `cached_verified` | `True` | 4 | `Stage41_fresh_self_gated_endpoint_candidate` | `True` | same-protocol cached verified architecture audit; current positive path is protected endpoint/full-waypoint dynamics |

## Retrain Boundary

- all components retrained inside Stage42-D: `False`
- reason: Stage42-D fresh-runs the safety/floor/full-waypoint ablations and verifies prior Stage30/41 retrained or architecture ablations by hash/source. It does not retrain every JEPA/Transformer/history/scene/goal component again in this command.
- source policy: fresh_run rows are recomputed this stage; cached_verified rows are old evidence with schema/hash/no-leakage provenance; not_run rows must remain not_run.

## Summary

- Stage42-B verdict: `stage42_b_external_validation_pass_protected_neural_not_ungated`
- Stage42-C verdict: `stage42_c_full_waypoint_dynamics_pass`
- required ablation coverage gate: `True`
- same-protocol architecture ablation gate: `True`
- protected endpoint all/t50/hard/easy: `0.2103` / `0.1365` / `0.2038` / `-0.1451`
- protected full-waypoint all/t50/t100diag/hard/easy: `0.1858` / `0.1480` / `0.2286` / `0.1952` / `-0.0000`

## Verdict

`stage42_d_causal_ablation_evidence_pass_with_retrain_boundary` (12 / 12)
