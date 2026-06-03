# Stage43-CP T100 Source/Scene Support Split Repair

- source: `fresh_stage43_cp_t100_source_scene_support_split_repair`
- result_source: `fresh_agent_disjoint_source_scene_supported_t100_split_manifest`
- verdict: `stage43_cp_t100_source_scene_support_split_ready`
- gate: `13 / 13`
- assignment hash: `a6b33fa427f68e5765fc92cfc6666dfde8252e414d23c95c46a4d661700c97d2`
- new model training run: `False`
- deployable policy changed: `False`

## Why I Built This

- Stage43-CO proved the current source-level heldout split has zero exact source/scene support for t100.
- This manifest builds a separate agent-disjoint source/scene-supported protocol for future t100 training/evaluation.
- It is not a cross-source generalization result; the source and scene overlap is intentional and reported.

## Split Summary

| split | rows | domains | sources | scenes | source-agents | horizons |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| train | 200442 | `{'ETH_UCY': 88073, 'TrajNet': 72566, 'UCY': 39803}` | 18 | 11 | 3261 | `{'10': 62746, '25': 55643, '50': 48573, '100': 33480}` |
| val | 67862 | `{'ETH_UCY': 30450, 'TrajNet': 24162, 'UCY': 13250}` | 18 | 11 | 1086 | `{'10': 21157, '25': 18799, '50': 16462, '100': 11444}` |
| test | 69687 | `{'ETH_UCY': 32275, 'TrajNet': 24162, 'UCY': 13250}` | 18 | 11 | 1086 | `{'10': 21651, '25': 19289, '50': 16927, '100': 11820}` |

## T100 Support

- test t100 rows: `11820`
- source-supported test t100 rows: `11724`
- scene-supported test t100 rows: `11820`
- source-or-scene-supported ratio: `100.00%`
- exact source-scene-supported ratio: `99.19%`
- unsupported test t100 rows: `0`

## Leakage Boundary

- row disjoint: `True`
- source-agent disjoint: `True`
- source overlap counts: `{'train_val': 18, 'train_test': 18, 'val_test': 18}`
- scene overlap counts: `{'train_val': 11, 'train_test': 11, 'val_test': 11}`
- source/scene overlap is intentional for this support protocol and is not a cross-source generalization split.
- no future endpoint/waypoint input, central velocity input, test endpoint goals, or test statistics normalization is constructed.

## Interpretation

- This gives a legal next path for t100 learning with validation support at source/scene level.
- It does not replace the stricter heldout generalization result from Stage43-CO, where t100 remains floor-only.
- The next step is to rebuild a light supervised cache on this protocol and evaluate whether t100 can improve without easy harm.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
