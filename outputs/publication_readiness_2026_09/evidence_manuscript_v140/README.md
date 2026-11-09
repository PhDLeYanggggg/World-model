# Evidence manuscript v140

The new supplement section tests harm prediction on actually
nominated paths and separates the candidate-menu limit from
the edit-count limit. Neither fitted policy qualifies. The
eight-page main paper is unchanged from
[v139](../evidence_manuscript_v139/README.md).

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v140
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v140.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `supplement/supplement.pdf` (41 pages) | `84b065123805319e1108fba204f7cd837488d7382aa290a5bf881d0038315290` |
| `supplement_source.zip` | `7a5ff6e09ef79d6dfc4a64f1a4ea9b0a9c9ebda2b039222919cef8a409364c40` |

All 41 page renders match a clean source-archive build at 85 DPI
with the separately supplied style. The revised page was inspected.
This is a working supplement, not an independent external result.
PDF, source ZIP, and model weights remain local.

[Experiment](../moving_cv_selected_harm_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
