# Evidence manuscript v150

The paper records the negative fixed-slot spline result. The supplement
reports nested calibration, three seeds, matched slots and displacement,
and a paired location bootstrap. The four guarded substitutions do not
establish a joint-choice advantage. This is a working draft, not an
independent result or a submission-ready package.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v150
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v150.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v150.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` (8 pages) | `a9f3102f5ad938416dfcdff2bb5715813716f14cb5495f8d3cf7487c30214ca5` |
| `supplement/supplement.pdf` (45 pages) | `e8d77020cc01f8be3172a4c59486e66dab82d47022d75ede1fcdd47e2f0c5d0d` |
| `main_source.zip` | `ecba33aa45571b3765e269bed85610a39535fe6d4a55f592afa1b0aec2adf563` |
| `supplement_source.zip` | `605c79430d948fe318dcc2363b896d5ca3df6468761c8d8bf45c634df8ad9963` |

All 53 pages match clean builds from the two archives at 85 DPI with
the separately supplied style. The changed main and supplement pages
were visually checked. The archives reproduce the working manuscript,
not the model training or independent confirmation. PDF and ZIP copies
remain local.

[Experiment](../opendd_nested_spline_choice_v2/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md)
