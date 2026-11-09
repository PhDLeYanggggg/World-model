# Evidence manuscript v149

The main paper and supplement add the matched-budget OpenDD direct
forecasts, leave-smallest-location sensitivity, and fixed-slot risk
comparison. The positive full-roster forecast contrast does not
qualify a replacement policy or independent transfer claim.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v149
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v149.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v149.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` (8 pages) | `0cd32544c5bba071c1f66e3ecba200b82768b79657aa77bb1c94d234be2663ca` |
| `supplement/supplement.pdf` (44 pages) | `90d3b479376014a2a4e3dfdaacb544c5ca776ef70d37086a2c32de41988979cb` |
| `main_source.zip` | `e5c6cb61f3929c5529a27967515f642ffd5bc3f3550efd92492d2bd03c7ec997` |
| `supplement_source.zip` | `6f6061ce09b89156e1fddfb9a1bd43df2de9af5dca21146c61f25466b288e916` |

All 52 pages match clean builds from the source archives at 85 DPI
with the separately supplied CVPR style. The modified sections
were visually checked. Source archives reproduce the document,
not the training or independent confirmation. PDFs and large
assets stay local.

[Experiment](../opendd_matched_spline_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md)
