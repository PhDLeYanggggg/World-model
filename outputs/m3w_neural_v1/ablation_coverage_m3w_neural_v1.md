# M3W-Neural v1 Ablation Coverage Audit

- source: `fresh_run`
- coverage gate: `True`
- missing: `[]`
- partial: `[]`
- cross-protocol limitations: `[]`
- same-protocol negative architecture evidence: `['no_jepa', 'no_transformer']`
- same-protocol architecture ablation gate: `True`
- Stage5C executed: `False`
- SMC enabled: `False`

| ablation | status | source | interpretation |
| --- | --- | --- | --- |
| `no_history` | `complete` | `outputs/stage41_fresh_confirmation/stage41_teacher_guided_evidence.json` | Masks history/static causal feature group after policy freeze; proves coverage for no-history/static ablation, not a claim that every history-derived scalar is useless. |
| `no_neighbor` | `complete` | `teacher_guided_evidence + group_consistency_evidence` | Neighbor/interaction masking is audited; group/neighbor features are especially important for the safety head and t100/hard slices. |
| `no_scene_goal` | `complete` | `stage41_teacher_guided_evidence + stage41_route_physical_policy_integration` | Scene/goal proxy coverage exists. Current deployable trajectory path keeps route/physical mostly diagnostic; route/physical heads are not main trajectory deployment claims. |
| `no_interaction` | `complete` | `stage41_teacher_guided_evidence + stage41_group_consistency_evidence` | Interaction/group-consistency features have explicit ablations and are necessary for guarded deployment; without them raw neural remains less safe. |
| `no_jepa` | `complete_with_same_protocol_negative_evidence` | `stage41_jepa_deployment_decision + stage41_neural_architecture_ablation_audit + stage30_m3w_verified/retrained_ablation_fresh` | JEPA is explicitly disabled from the deployable path because audited JEPA variants were non-collapse but did not give deployable downstream lift. Stage41 now also records same-protocol pure-Transformer/no-JEPA attempts as negative or fallback-only, so the current positive path is protected endpoint neural dynamics rather than JEPA/Transformer purity. |
| `no_transformer` | `complete_with_same_protocol_negative_evidence` | `stage41_neural_architecture_ablation_audit + stage30_m3w_verified/retrained_ablation_fresh + Stage41 JEPA-only diagnostics` | Stage41 same-protocol JEPA-only/no-Transformer attempts are negative or unsafe, so no-Transformer is covered as negative architecture evidence. This is not a claim that JEPA contributes to the deployable path; it is why JEPA remains diagnostic-only. |
| `no_fallback` | `complete` | `teacher_guided_evidence + pure_ucy_neural_statistical_evidence + all_agent_composite_world_state` | No-fallback neural often improves hard/all raw error but catastrophically damages easy cases; fallback is required for deployability. |

## Important Boundary

This audit keeps the evidence traceable. no-JEPA and no-Transformer are now covered by same-protocol negative architecture evidence; any future cross-protocol limits must remain explicit.
The no-fallback evidence remains negative for deployment safety; Stage37/teacher fallback remains required.
- no leakage: `{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'test_endpoint_goals': False, 'central_velocity': False, 'stage5c_executed': False, 'smc_enabled': False}`
- claim boundary: `{'ablation_coverage_not_new_training': True, 'cross_protocol_ablations_are_limitations': [], 'not_true_3d': True, 'not_foundation': True, 'not_metric_or_seconds': True}`
