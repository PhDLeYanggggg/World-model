# Stage43-DI T100 Support-Aware Bounded-Alpha Distilled Head

- source: `fresh_stage43_di_t100_support_aware_bounded_alpha_distilled_head`
- result_source: `fresh_torch_support_aware_bounded_alpha_distilled_t100_head`
- verdict: `stage43_di_t100_support_aware_distilled_head_safe_but_no_lift_diagnostic`
- gate: `15 / 15`
- seeds: `[4323, 4331, 4337]`
- deploy on current heldout t100: `False`

## Aggregate

- mean t100 improvement: `0.001570`
- mean min-without-group t100: `0.000824`
- all min-without-group positive: `True`
- max easy degradation: `0.000000`
- mean switch rate: `0.139400`
- beats DH t100 mean: `False`
- beats DE t100 mean: `False`

## Per Seed

| seed | t100 | min-without-group | easy degradation | switch rate | teacher switch | bootstrap low | best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4323` | `0.001129` | `0.000713` | `0.000000` | `0.087100` | `0.029214` | `0.000958` | `4` |
| `4331` | `0.001903` | `0.001119` | `0.000000` | `0.179900` | `0.021851` | `0.001608` | `4` |
| `4337` | `0.001680` | `0.000642` | `0.000000` | `0.151200` | `0.024071` | `0.001402` | `3` |

## Interpretation

- DH fixed the DF head's selection gap by reranking existing candidates.
- DI retrains a new bounded-alpha head using DH support-aware policies as train-row teacher labels, then uses the same support-aware validation selector.
- Checkpoints are written locally for replay, but not committed.
- This is not deployed unless it beats the stronger DE bounded policy while preserving every seed/group/easy gate.
- Future waypoints are labels/eval only; inference inputs remain causal.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
