# Stage43-CU T100 Residual Admissibility Statistical Confirmation

- source: `fresh_stage43_cu_t100_residual_admissibility_statistical_confirmation`
- result_source: `fresh_torch_t100_residual_admissibility_multiseed_confirmation`
- verdict: `stage43_cu_t100_admissibility_multiseed_confirmed_tiny_positive`
- gate: `12 / 12`
- seeds: `[4323, 4331, 4337]`
- deploy on current heldout t100: `False`

## Aggregate

- mean t100 improvement: `0.001174`
- min t100 improvement: `0.000841`
- mean hard/failure improvement: `0.001174`
- max easy degradation: `0.000000`
- mean switch rate: `0.066200`
- all seed bootstrap low positive: `True`

## Per Seed

| seed | t100 | hard/failure | easy degradation | switch rate | bootstrap low |
| --- | ---: | ---: | ---: | ---: | ---: |
| `4323` | `0.000841` | `0.000841` | `0.000000` | `0.067000` | `0.000575` |
| `4331` | `0.001180` | `0.001180` | `0.000000` | `0.061600` | `0.000990` |
| `4337` | `0.001501` | `0.001501` | `0.000000` | `0.070000` | `0.001299` |

## Interpretation

- This confirms whether the Stage43-CT tiny supported-protocol t100 lift survives seed variation and bootstrap.
- The effect remains a supported-protocol diagnostic, not a current heldout deployment change.
- Future endpoints/full waypoints are labels only; inference inputs remain causal.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
