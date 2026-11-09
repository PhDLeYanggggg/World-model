# Evidence manuscript v139

The supplement adds the amplitude response experiment: better
prediction of the gain curve did not improve selection at matched
counts or matched displacement. No candidate qualified under the
unchanged rule. The eight-page main text is unchanged from v138.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v139
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v139.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v139.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` (8 pages) | `ca7dce0366d5d9ef55b6edd4c4e1589fa1c60e3f1c42f5a851ebedb3587740dc` |
| `supplement/supplement.pdf` (41 pages) | `f91af0bda83bb8c041528cf2fa1cd7bae490313df135cd80a81655303139f7db` |
| `main_source.zip` | `913d343dcfbffc4c93bdbbcdb99ee6600999e446ce07dad970006e389639dc32` |
| `supplement_source.zip` | `ffab3072a8a5de6f65a5f2014df700f4534f52853fb2bb65517f20b234d5b7fa` |
| `../moving_cv_amplitude_response_v1/anonymous_amplitude_response.zip` | `32a3631ff789d233c6e0493339c934462a8dd90403487bf39e75c8392f328a37` |

All 49 page renders match clean source-archive builds at 85 DPI
with the separately supplied style. The changed pages were inspected.
The anonymous archive verifies the scalar policy table after clean
extraction. PDFs, archives and checkpoints remain local.

[Experiment](../moving_cv_amplitude_response_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
