# Evidence Manuscript v81

Integrates the [donor-selected stopping result](../one_edit_stopping_v1/README.md).
The main text separates exact one-edit capacity from the measured
loss at matched intervention counts. The supplement records the
full-roster trade-off, fixed delay and secondary donor guard.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v81
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v81.py main --source-zip "$out/manuscript_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v81.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The 8-page main and 21-page supplement built from the workspace
and from clean source archives with the same official CVPR style.
All 29 rendered pages matched at 85 DPI. The changed pages, figures,
disclosures and references were visually checked. PDFs and archives
remain local.

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` | `fc79b615e9975a499488041aa51c62a15997dcd70b1e29294f4acc0dfeb58988` |
| `manuscript_source.zip` | `74f6e588f8a8db5bdf719911e9f47fcf179badfe1dac25413e0e1e0c4d099aea` |
| `supplement/supplement.pdf` | `acac3f5067eb09d16ee618af79ff66be0815a09bd6488eea0cda878043ee8589` |
| `supplement_source.zip` | `ee4797cf4b4f8e20c674f05592b6aba33e87c29cbb3415204c43a1277486f0b5` |

Development evidence only. No independent confirmation, submission
or readiness claim.
