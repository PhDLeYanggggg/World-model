# Evidence manuscript v159

This working revision adds the frozen, one-shot WEPDTOF test to
the abstract, limitations and supplement. It distinguishes strict
abstention from a fixed-count utility comparison on 13 original
source groups. The latter provides no evidence that joint choice
improves the independent control, and no recording-seed view
passes both original risk and utility against both motion baselines.
This is a negative external test, not a certified learned deployment
or a submission-ready result. DroneCrowd confirmation stays sealed.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v159
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v159.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v159.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` (8 pages) | `b14575bf97cad4378825db03b11fe14fe3e59082e40def807b379ce60dcda89b` |
| `supplement/supplement.pdf` (49 pages) | `c2bffef9900a2e177b0c5510df59b50c7cbbabc72fb880aebc4bc61df03b4a02` |
| `main_source.zip` | `3d3d2cf6740a09fa0d490e0e9135f2be99590899b4742314b17abd41f063ee1c` |
| `supplement_source.zip` | `54ce781224a99add8c012381008fcefad55837b1083459c65975c5f6c3333338` |

A clean source-archive build with the CVPR style supplied separately
matched all 57 rendered pages at 85 DPI. The changed pages were
visually checked. No registration or submission was performed.
PDFs, ZIPs, raw camera data, checkpoints and track caches stay local.

[Experiment](../wepdtof_external_transfer_v1/README.md) |
[Chinese reproduction](../../../reproducibility/M3W_CURRENT_ZH.md)
