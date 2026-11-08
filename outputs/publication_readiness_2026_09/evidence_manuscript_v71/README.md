# Evidence Supplement v71

This version records the [matched recipient experiment](../recipient_slot_v1/README.md)
alongside the same-radius shape controls. The intervention times and radii
limit easy-reference support even with label-aware recipient assignment.
The causal recipient head does not improve on the fixed-roster control.
The [v70 main manuscript](../evidence_manuscript_v70/README.md) is unchanged.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v71
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v71.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The supplement has 19 pages. The direct build and the independently
unpacked source build with the same official CVPR style render identically
at 85 DPI. The added paragraph was visually inspected. Local files:

| Artifact | SHA-256 |
|---|---|
| `supplement/supplement.pdf` | `ba3f4659cd8377d34674f6e632a927f68e5513f7e3d7786ef5dcafd40317addb` |
| `supplement_source.zip` | `e0ab36fe72312501cf9d7bc30f88113504ecd8824e7bdc7b613d33bbb9d8d05c` |

These are exposed development experiments, not independent confirmation.
The original risk rule, deployment decision and sealed DroneCrowd roster
are unchanged. Nothing was enrolled or submitted.
