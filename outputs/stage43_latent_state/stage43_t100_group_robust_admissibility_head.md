# Stage43-DA T100 Group-Robust Admissibility Head

- source: `fresh_stage43_da_t100_group_robust_admissibility_head`
- result_source: `fresh_torch_group_robust_t100_admissibility_head`
- verdict: `stage43_da_t100_group_robust_head_positive_but_not_policy_best`
- gate: `14 / 14`
- seeds: `[4323, 4331, 4337]`
- deploy on current heldout t100: `False`

## Aggregate

- mean t100 improvement: `0.001379`
- min t100 improvement: `0.001203`
- mean min-without-group t100: `-0.000305`
- max easy degradation: `0.000000`
- mean switch rate: `0.069067`
- all bootstrap lows positive: `True`
- beats CZ t100 mean: `False`
- beats CZ min-without-group mean: `False`
- CZ robust t100 mean: `0.001841`
- CZ robust min-without-group mean: `0.000995`

## Per Seed

| seed | t100 | min-without-group | easy degradation | switch rate | bootstrap low | best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `4323` | `0.001229` | `-0.000222` | `0.000000` | `0.082000` | `0.001043` | `5` |
| `4331` | `0.001203` | `-0.000171` | `0.000000` | `0.050300` | `0.000897` | `3` |
| `4337` | `0.001703` | `-0.000520` | `0.000000` | `0.074900` | `0.001344` | `4` |

## Interpretation

- This trains a fresh t100 admissibility head with source/scene/domain-balanced sample weights and a support penalty.
- Checkpoint selection uses the leave-group-out validation objective introduced in Stage43-CZ.
- Future waypoints are labels/eval only; inference inputs are causal CS diagnostics, latent state, history/goal/baseline features, and split metadata.
- This is still diagnostic and does not deploy current heldout t100.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
