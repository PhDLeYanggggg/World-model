# Stage43-DE T100 Policy-Distilled Alpha-Stability Policy

- source: `fresh_stage43_de_t100_policy_distilled_alpha_stability_policy`
- result_source: `fresh_bounded_alpha_policy_selection_on_dc_head`
- verdict: `stage43_de_t100_alpha_stability_policy_repairs_group_fragility_diagnostic`
- gate: `15 / 15`
- deploy on current heldout t100: `False`

## Aggregate

- original DC t100 mean: `0.002167`
- bounded t100 mean: `0.001860`
- DD guarded t100 mean: `0.002176`
- bounded min-without-group mean: `0.000920`
- all bounded min-without-group positive: `True`
- bounded easy degradation max: `0.000000`
- selected variants: `['alpha_cap_0_75', 'alpha_cap_0_75', 'alpha_cap_0_75']`
- selected alphas: `[0.75, 0.75, 0.75]`
- repairs DD seed fragility: `True`
- beats CZ t100 mean: `True`
- mean t100 tradeoff vs DD: `-0.000316`

## Per Seed

| seed | variant | alpha | bounded t100 | bounded min-without | easy | switch | delta t100 vs original |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `4323` | `alpha_cap_0_75` | `0.75` | `0.002133` | `0.001049` | `0.000000` | `0.146400` | `-0.000423` |
| `4331` | `alpha_cap_0_75` | `0.75` | `0.001780` | `0.001189` | `0.000000` | `0.150800` | `-0.000497` |
| `4337` | `alpha_cap_0_75` | `0.75` | `0.001666` | `0.000521` | `0.000000` | `0.094200` | `0.000000` |

## Interpretation

- DD showed that the policy-distilled head's remaining problem is seed-level group fragility, especially full alpha=1.0 intervention on one heldout scene slice.
- This step applies a fixed bounded-intervention rule: validation may choose the best safe policy only among alpha <= 0.75 candidates.
- The result repairs the DD seed-level negative min-without-group slice, but it trades away some of DC/DD's mean t100 gain.
- I am keeping this diagnostic rather than deploying it: the current long-horizon head still needs stronger training evidence before replacing the protected floor.
- Future waypoints remain labels/eval only; inference inputs remain causal.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
