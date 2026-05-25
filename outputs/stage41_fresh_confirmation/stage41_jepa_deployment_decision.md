# Stage41 JEPA Deployment Decision

- source: `fresh_run`
- decision: `disable_jepa_in_deployable_path`
- attempts audited: `7`
- non-collapse attempts: `7`
- deployable positive attempts: `0`
- Stage5C executed: `False`
- SMC enabled: `False`

| Attempt | Non-collapse | Deployable lift | Key lifts |
| --- | --- | --- | --- |
| SAM-JEPA-2.5D representation pretraining | `True` | `False` | `{'failure_auroc_lift': -0.012096774193548376}` |
| WAM-style JEPA probe | `True` | `False` | `{'failure_auroc_lift': -0.08754208754208748, 'selector_t50_lift': 0.0}` |
| Stage23 SDD JEPA | `True` | `False` | `{'selector_probe_lift': 0.0, 'failure_predictor_lift': 0.0, 'hard_failure_correction_lift': 0.0, 't50_lift': 0.0}` |
| Stage24 SDD JEPA | `True` | `False` | `{'selector_probe_lift': 0.0, 'failure_predictor_lift': 0.0, 'hard_failure_correction_lift': 0.0, 't50_lift': 0.0}` |
| Stage39 JEPA auxiliary representation | `True` | `False` | `{'failure_auroc_lift': -0.4285583008392251, 'failure_auroc_base': 0.9880898558164535, 'failure_auroc_with_jepa': 0.5595315549772284}` |
| Stage40 jepa_aux_candidate_ranker | `True` | `False` | `{'all_improvement': -0.31813566333776033, 't50_improvement': -0.36523721047520064, 'hard_failure_improvement': -0.3121845224802864, 'easy_degradation': 0.7812278656398819, 'switch_rate': 0.02}` |
| Stage40 hybrid_moe_deeper_ranker | `True` | `False` | `{'all_improvement': -0.22194761788310857, 't50_improvement': -0.6152104089274417, 'hard_failure_improvement': -0.18030445536332285, 'easy_degradation': 1.2056777651489945, 'switch_rate': 0.02}` |

Conclusion: JEPA remains a diagnostic representation path only. The deployable M3W-Neural v1 path uses the Stage37 floor plus neural group-consistency safety/gain heads, not JEPA. This avoids overstating non-collapse JEPA as downstream world-model contribution.
