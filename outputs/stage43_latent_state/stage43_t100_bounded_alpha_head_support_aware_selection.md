# Stage43-DH T100 Bounded-Alpha Head Support-Aware Selection

- source: `fresh_stage43_dh_t100_bounded_alpha_head_support_aware_selection`
- result_source: `fresh_support_aware_validation_selection_over_df_head`
- verdict: `stage43_dh_t100_support_aware_selection_repairs_df_group_fragility_diagnostic`
- gate: `14 / 14`
- deploy on current heldout t100: `False`

## Aggregate

- selected t100 mean: `0.001640`
- selected min-without-group t100 mean: `0.000264`
- all min-without-group positive: `True`
- max easy degradation: `0.000000`
- support-safe candidate min count: `212`
- delta t100 vs DF legacy mean: `0.000167`
- delta min-without-group vs DF legacy mean: `0.000058`

## Per Seed

| seed | selected t100 | selected min | easy | switch | delta t100 | delta min | policy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `4323` | `0.001859` | `0.000675` | `0.000000` | `0.151200` | `0.000286` | `0.000907` | `a=0.75,g=0.2,h=0.35,d=0.0` |
| `4331` | `0.001298` | `0.000047` | `0.000000` | `0.135500` | `0.000000` | `0.000000` | `a=0.75,g=0.5,h=0.35,d=0.0` |
| `4337` | `0.001762` | `0.000069` | `0.000000` | `0.119700` | `0.000215` | `-0.000731` | `a=0.75,g=0.2,h=0.35,d=0.0` |

## Interpretation

- DG showed the DF head had safe candidates but validation picked a heldout-fragile candidate.
- This reranks DF head candidates on validation with t100, min-without-group, support coverage, concentration, and only a light switch penalty.
- Test rows are used once for evaluation; no threshold is chosen from test.
- This repairs the DF head's group-fragility symptom, but remains diagnostic because it does not beat the stronger DE bounded policy on mean t100.
- Future waypoints are labels/eval only; inference inputs remain causal.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
