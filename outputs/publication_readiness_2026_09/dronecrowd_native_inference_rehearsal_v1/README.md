# A Shared Inference Entry Point For Native 8/12 Forecasting

The external comparison now has one past-only inference entry point for
six causal baselines, three frozen self-kinematic fits and the 18
checkpoint comparators. It keeps target agent IDs and origins aligned,
uses the original complete-history neighbor rule, and emits relative
image-coordinate paths. No coordinate or time rescaling is performed.
CV remains the risk reference. OLS4 remains the comparator selected
on producer fitting data, not an external winner.

## What Replayed Exactly

| Check | Scope |
|---|---:|
| Geometry rebuilt from the eight observed positions | 125,540 rows, 1,613 queries, 34 development recordings |
| Self-kinematic paths | All 125,540 rows for each of seeds 17, 29 and 43 (376,620 paths) |
| EqMotion and Past Context Transformer (PCT) | 1,024 original probe paths each for 18 checkpoints (18,432 paths) |

The geometry and paths match the registered caches element for element.
The stored comparator probe batch boundaries were preserved. PCT is
the project's own causal Transformer comparator, not a published result.
The [EqMotion](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_CVPR_2023_paper.html)
comparison uses the project's frozen fixed-head adaptation, not the
paper's original multimodal scoring protocol. No optimizer step was run.
The previously verified image-screen receipts cover these checkpoint
training sources; they do not certify geographic independence.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_native_inference_rehearsal.py run
.venv-pytorch/bin/python scripts/run_m3w_native_inference_rehearsal.py replay
```

The first native arm64 run took 43 seconds and peaked at about 2.1 GB
of resident memory, with four compute threads and zero loader workers.
The replay binds the same [analysis](analysis.json). Sixteen targeted
tests passed, including future-field rejection and incomplete-history
handling. Code and receipts are public; local data and checkpoints are not.

## Remaining External Work

This command deliberately reads development prefixes only. It does not
open DroneCrowd inputs or future labels, assign the proposed scientific
roles, freeze a controller or reference cuts, or score outcomes. The
already-completed external input scan and grouping are
[separate](../dronecrowd_blind_preflight_v1/README.md).
The unresolved choice is external calibration on 15 groups with
confirmation on 30, or frozen transfer on all 45. Confirm that choice
and freeze the controller and cuts before external inference and scoring.
The original 2% and reference-support rules are unchanged. This is an
engineering result, not independent confirmation or a deployment change.
No metric, seconds, true 3D, foundation, Stage5C or SMC claim.
