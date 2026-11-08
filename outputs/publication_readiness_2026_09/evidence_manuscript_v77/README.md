# Evidence Supplement v77

Adds the equal-radius [turn comparison and exact one-edit bound](../sustained_turn_v2/README.md).
This result rules out a single-edit solution for part of the existing
development roster even with arbitrary candidate geometry. It does not
establish a new predictor, a risk guarantee or independent confirmation.
The [main paper v73](../evidence_manuscript_v73/README.md) is unchanged.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v77
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v77.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The 20-page supplement built from the workspace and from its freshly
extracted archive using the same official CVPR style. All 20 renders
matched at 85 DPI. The changed pages, disclosure and references were
visually checked. PDF and source archive remain local.

| Artifact | SHA-256 |
| --- | --- |
| `supplement/supplement.pdf` | `8607436e70f1049687a4578e23fd1dba657f0c1aa0701ac23f989c60828bf8da` |
| `supplement_source.zip` | `43cdfbcd84527dca1130f422957cae05f6b941af76bf8aa6f08981d386372768` |

No submission or claim of readiness.
