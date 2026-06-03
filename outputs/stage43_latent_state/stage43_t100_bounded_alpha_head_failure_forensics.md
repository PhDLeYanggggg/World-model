# Stage43-DG T100 Bounded-Alpha Head Failure Forensics

- source: `fresh_stage43_dg_t100_bounded_alpha_head_failure_forensics`
- result_source: `fresh_bounded_alpha_head_selection_forensics`
- verdict: `stage43_dg_t100_bounded_alpha_head_forensics_selection_gap_identified`
- gate: `13 / 13`
- failure root: `validation_group_risk_selection_gap`
- deploy on current heldout t100: `False`

## Aggregate

- selected t100 mean: `0.001472`
- selected min-without-group mean: `0.000205`
- positive candidate exists all seeds: `True`
- selected group-positive all seeds: `False`
- selection misses safe candidate: `True`
- positive group candidate count min: `164`
- oracle min gap mean: `0.000515`
- oracle t100 gap mean: `0.000180`

## Per Seed

| seed | candidates | positive candidates | selected t100 | selected min | oracle min gap | oracle t100 gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `4323` | `278` | `164` | `0.001573` | `-0.000232` | `0.000976` | `0.000286` |
| `4331` | `277` | `253` | `0.001298` | `0.000047` | `0.000568` | `0.000020` |
| `4337` | `276` | `175` | `0.001547` | `0.000801` | `0.000000` | `0.000234` |

## Interpretation

- This is diagnostic forensics over the already-trained DF head, not a deployment policy.
- The test oracle is used only to explain why DF failed its gate; no threshold or policy is promoted from test.
- If positive candidates exist but validation selects a fragile one, the next fix is a validation group-risk objective or better support split, not another blind head retrain.
- If no positive candidates exist, the next fix must change the head/data/latent target.
- Future waypoints remain labels/eval only; inference inputs remain causal.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
