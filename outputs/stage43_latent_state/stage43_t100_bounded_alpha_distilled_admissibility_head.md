# Stage43-DF T100 Bounded-Alpha Distilled Admissibility Head

- source: `fresh_stage43_df_t100_bounded_alpha_distilled_admissibility_head`
- result_source: `fresh_torch_bounded_alpha_policy_distilled_t100_head`
- verdict: `stage43_df_t100_bounded_alpha_distilled_head_incomplete`
- gate: `14 / 15`
- seeds: `[4323, 4331, 4337]`
- deploy on current heldout t100: `False`

## Aggregate

- mean t100 improvement: `0.001472`
- mean min-without-group t100: `0.000205`
- all min-without-group positive: `False`
- max easy degradation: `0.000000`
- mean switch rate: `0.100567`
- beats DE t100 mean: `False`
- beats DE min-without-group mean: `False`

## Per Seed

| seed | t100 | min-without-group | easy degradation | switch rate | teacher switch | bootstrap low | best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4323` | `0.001573` | `-0.000232` | `0.000000` | `0.086900` | `0.021000` | `0.001319` | `3` |
| `4331` | `0.001298` | `0.000047` | `0.000000` | `0.135500` | `0.024470` | `0.001023` | `3` |
| `4337` | `0.001547` | `0.000801` | `0.000000` | `0.079300` | `0.014804` | `0.001310` | `5` |

## Interpretation

- This trains a new head from the DE bounded-alpha policy instead of only adding an outer alpha cap at deployment time.
- Teacher labels are built on train rows using the DC teacher head and DE validation-selected bounded-alpha policies.
- Validation still selects policy/checkpoint and test is evaluated once; checkpoints are written locally but not committed.
- Future waypoints are labels/eval only; inference inputs remain causal.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
