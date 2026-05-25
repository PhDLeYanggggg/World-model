# Stage41 Neural Architecture Ablation Audit

- source: `fresh_run`
- same-protocol architecture ablation gate: `True`
- best protected architecture: `Stage41_fresh_self_gated_endpoint_candidate`
- transformer-only deployable: `False`
- JEPA-only deployable: `False`
- hybrid deployable: `False`
- mixture selector deployable: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

## Architecture Groups

| group | attempted | best candidate | best status | all | t+50 | t+100 diag | hard/failure | easy | positive domains |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `transformer_only` | `True` | `Stage41_conformal_safety_head_transformer` | `safe_fallback_only_no_lift` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| `jepa_only` | `True` | `Stage41_jepa_auxiliary_representation` | `negative_or_unsafe` | -0.026777069471576986 | -0.015065713704299988 | 0.0 | -0.024300942834727834 | 0.026903930506348317 | 0 |
| `hybrid_jepa_transformer` | `True` | `Stage41_easy_guard_distilled_hybrid` | `safe_fallback_only_no_lift` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| `mixture_selector` | `True` | `Stage41_mixture_of_experts_baseline_selector` | `safe_fallback_only_no_lift` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| `protected_neural_endpoint` | `True` | `Stage41_fresh_self_gated_endpoint_candidate` | `deployable_positive` | 0.41964214194307703 | 0.4061979981406123 | 0.45728888926366984 | 0.43608295876101655 | 0.0 | 3 |

## Interpretation

- Same-protocol pure Transformer/no-JEPA attempts are not deployable; they are negative or fallback-only evidence.
- Same-protocol JEPA-only/no-Transformer attempts are not deployable; JEPA remains diagnostic-only despite non-collapse in earlier stages.
- Same-protocol JEPA+Transformer hybrid attempts did not beat the Stage37 safety floor.
- The positive Stage41 path is protected endpoint neural dynamics under the Stage37/teacher safety floor, not an ungated JEPA/Transformer rollout.

## Claim Boundary

`{'not_true_3d': True, 'not_foundation': True, 'not_metric_or_seconds': True, 'same_protocol_architecture_audit_not_new_training': True, 'jepa_transformer_hybrid_positive_contribution_not_proven': True, 'protected_endpoint_neural_candidate_is_current_positive_neural_evidence': True, 'ungated_neural_still_not_deployable': True}`
