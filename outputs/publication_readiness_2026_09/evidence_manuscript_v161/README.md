# Evidence manuscript v161

I redrew the three main figures and the signed-coverage figure in
the supplement for print. The plots use grayscale, distinct markers,
thin axes and direct labels rather than headline-style titles. The
figures remain vector PDFs. Counts, source-level intervals,
evaluation rules and scientific claims are unchanged from v160.
The Figure 1 caption is shorter; the sample unit is unchanged.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v161
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v161.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v161.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` (8 pages) | `a779833975b096800b6d50b439edd10df918d7f1ad4305cedf2648a4588074e1` |
| `supplement/supplement.pdf` (49 pages) | `eabbf2c2c4fe757c3e5f9ebc912e2e4afd81e3c00c9cbf07b03a6af29cd57ed9` |
| `main_source.zip` | `5593855743b980514fc49e7b66e72098155dfc98460bbd8a600d42ef997b042b` |
| `supplement_source.zip` | `b16e63643e5a3b21ac66d2e7b0a8c334080870841e9883fa4426e00451fbb342` |

All 57 rendered pages matched a clean source-archive build at
85 DPI. I inspected the standalone figures and their page layout.
The PDFs, source ZIPs and image previews stay local. This is a
presentation revision, not a new experiment or a submission.

[Scientific revision v160](../evidence_manuscript_v160/README.md)
