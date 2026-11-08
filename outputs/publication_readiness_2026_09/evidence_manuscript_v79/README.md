# Evidence Manuscript v79

Integrates the [cross-window comparison](../cross_query_edit_v2/README.md)
into the main paper and supplement. The main text now distinguishes
label-aware timing capacity, a matched second-edit timing loss, and
the uncertain effect of coordination at the same query clocks.
The supplement retains the full-roster pass counts, the unmatched
incomplete schedules, and the first-edit-fixed oracle.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v79
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v79.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v79.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The 8-page main and 20-page supplement built from the working files
and fresh extracted source archives with the same official CVPR style.
All 28 page renders matched at 85 DPI. Main pages 7-8 and supplement
page 20 were visually checked. PDFs and ZIP files remain local.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `972545279766be1dbcf0a720af7319dcb4ed9c659259e109528641081dd13d54` |
| `manuscript_source.zip` | `3d601ec12b51f17a2de67944b3d0ec7ed7e78901428991bd3df9a238b1e02829` |
| `supplement/supplement.pdf` | `058a411cf27f8a58b332a44cd2e8e0316748af88b61cad141ecbd73a6282d147` |
| `supplement_source.zip` | `0441c45919bf7a68cfaf1e26f504be3097ef3322764f29f0686bfde2e8c76512` |

Development evidence only. No independent confirmation, submission
or claim of readiness.
