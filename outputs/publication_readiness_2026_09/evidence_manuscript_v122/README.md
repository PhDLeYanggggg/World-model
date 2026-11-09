# Evidence Manuscript v122

The main paper now distinguishes an accuracy comparator from a starting
forecast for deployment. On the exposed development roster, at most one
arbitrary edit per query cannot bring OLS4 within the original CV-relative
rule. The supplement includes the derivation and the matched source-deletion
experiment. This is a development result, not independent confirmation.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v122
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v122.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v122.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` (8 pages) | `afae6d45506e4ab388ac99191c306dbd927fe0585de79a243d9830aa77c83c3c` |
| `supplement/supplement.pdf` (35 pages) | `74e7713d13ac27502bc3b1788c852bc238d53d3d845b2a9388194896c84c63a5` |
| `main_source.zip` | `4ff3509a4f00f7251a78ab0a61df81ff91c3ffc27029903cb7b5b6d5fc02aa5a` |
| `supplement_source.zip` | `a2a446f840c1d4bea12cb82270f82f752fd8ff1fd2d31310499487d3e94a0514` |

[Experiments and optimistic risk bound](../moving_motion_cost_stability_v1/README.md).
All 43 pages matched clean source-archive builds at 85 DPI using the same
separately supplied CVPR style. The new main paragraph and supplement
derivation were visually checked. No
PDF, source archive, dataset or checkpoint is committed here. Not submitted.
