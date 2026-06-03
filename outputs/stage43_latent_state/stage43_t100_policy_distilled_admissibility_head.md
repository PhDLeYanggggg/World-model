# Stage43-DC T100 Policy-Distilled Admissibility Head

- source: `fresh_stage43_dc_t100_policy_distilled_admissibility_head`
- result_source: `fresh_torch_policy_distilled_t100_admissibility_head`
- verdict: `stage43_dc_t100_policy_distilled_head_beats_cz_diagnostic`
- gate: `13 / 13`
- seeds: `[4323, 4331, 4337]`
- deploy on current heldout t100: `False`

## Aggregate

- mean t100 improvement: `0.002167`
- mean min-without-group t100: `0.000058`
- max easy degradation: `0.000000`
- mean switch rate: `0.102333`
- mean teacher switch rate: `0.017633`
- beats DA t100 mean: `True`
- beats CZ t100 mean: `True`
- beats DA min-without-group mean: `True`
- beats CZ min-without-group mean: `False`

## Per Seed

| seed | t100 | min-without-group | easy degradation | switch rate | teacher switch | bootstrap low | best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4323` | `0.002556` | `-0.000491` | `0.000000` | `0.121200` | `0.014857` | `0.002122` | `5` |
| `4331` | `0.002277` | `0.000144` | `0.000000` | `0.091600` | `0.019661` | `0.001917` | `5` |
| `4337` | `0.001666` | `0.000521` | `0.000000` | `0.094200` | `0.018381` | `0.001399` | `3` |

## Interpretation

- This trains on CZ leave-group-out robust switch decisions rather than only generic gain/harm/delta labels.
- Teacher labels are built on train rows; validation still selects policy/checkpoint and test is evaluated once.
- Future waypoints are labels/eval only; inference inputs are causal CS diagnostics, latent state, history/goal/baseline features, and split metadata.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
