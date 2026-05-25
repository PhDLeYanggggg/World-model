# Stage42 Ablation Tables

| ablation | source | status | all | t50 | hard/failure | easy | interpretation |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `no_neural_tail_use_teacher_floor_only` | `fresh_run` | `positive_safe` | 0.2036 | 0.1312 | 0.1966 | -0.1445 | Removing the composite-tail neural safe switch leaves the Stage37/teacher floor; positive protected-minus-teacher deltas indicate neural tail contribution. |
| `no_safe_floor_use_ungated_endpoint_neural` | `fresh_run` | `negative_unsafe` | 0.2966 | 0.2152 | 0.3294 | 1.2459 | Ungated endpoint neural is a no-fallback safety ablation; it can improve raw all but is not deployable if easy degradation exceeds 2%. |
| `oracle_floor_vs_neural_diagnostic` | `fresh_run` | `positive_safe` | 0.4222 | 0.3452 | 0.4211 | -0.2999 | Diagnostic oracle uses future labels only to measure remaining headroom; it is not a deployable model. |
| `no_full_waypoint_sequence_use_endpoint_linear_bridge` | `fresh_run` | `positive_safe` | 0.2103 | 0.1365 | 0.2038 | -0.1451 | Endpoint-linear bridge removes the full-waypoint sequence model. delta_vs_reference is ablation-minus-protected: negative t50/t100 deltas mean the full-waypoint model helps those horizons, while positive all-delta means endpoint-linear remains stronger on all-ADE. |
| `no_safe_floor_use_ungated_full_waypoint` | `fresh_run` | `negative_unsafe` | 0.2966 | 0.2152 | 0.3294 | 1.2459 | Ungated full-waypoint neural is a no-fallback safety ablation; it remains diagnostic if easy degradation is unsafe. |
| `no_composite_tail_use_teacher_linear_bridge` | `fresh_run` | `positive_safe` | 0.2036 | 0.1312 | 0.1966 | -0.1445 | Teacher linear bridge is the pre-composite floor in waypoint space; protected full-waypoint must improve without easy harm. |
| `no_history` | `cached_verified` | `complete` | n/a | n/a | n/a | n/a | Masks history/static causal feature group after policy freeze; proves coverage for no-history/static ablation, not a claim that every history-derived scalar is useless. |
| `no_neighbor` | `cached_verified` | `complete` | n/a | n/a | n/a | n/a | Neighbor/interaction masking is audited; group/neighbor features are especially important for the safety head and t100/hard slices. |
| `no_scene_goal` | `cached_verified` | `complete` | n/a | n/a | n/a | n/a | Scene/goal proxy coverage exists. Current deployable trajectory path keeps route/physical mostly diagnostic; route/physical heads are not main trajectory deployment claims. |
| `no_interaction` | `cached_verified` | `complete` | n/a | n/a | n/a | n/a | Interaction/group-consistency features have explicit ablations and are necessary for guarded deployment; without them raw neural remains less safe. |
| `no_jepa` | `cached_verified` | `complete_with_same_protocol_negative_evidence` | n/a | n/a | n/a | n/a | JEPA is explicitly disabled from the deployable path because audited JEPA variants were non-collapse but did not give deployable downstream lift. Stage41 now also records same-protocol pure-Transformer/no-JEPA attempts as negative or fallback-only, so the current positive path is protected endpoint neural dynamics rather than JEPA/Transformer purity. |
| `no_transformer` | `cached_verified` | `complete_with_same_protocol_negative_evidence` | n/a | n/a | n/a | n/a | Stage41 same-protocol JEPA-only/no-Transformer attempts are negative or unsafe, so no-Transformer is covered as negative architecture evidence. This is not a claim that JEPA contributes to the deployable path; it is why JEPA remains diagnostic-only. |
| `no_fallback` | `cached_verified` | `complete` | n/a | n/a | n/a | n/a | No-fallback neural often improves hard/all raw error but catastrophically damages easy cases; fallback is required for deployability. |

## Boundary

Stage42-D fresh-runs safety/floor/full-waypoint ablations and cached-verifies prior Stage30/41 component ablation evidence. It does not complete all-component retraining inside Stage42-D.
