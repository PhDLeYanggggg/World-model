# Evidence manuscript v133

The main paper and supplement now distinguish verified training
teachers, held development choices, and hindsight capacity under
the exact frozen query budgets. The new experiment does not qualify
a strategy or support an independent generalization claim.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v133
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v133.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v133.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` (8 pages) | `05821185d8ef895d998bb665d990d266d31456833411bb69451b731b271ac5f6` |
| `supplement/supplement.pdf` (39 pages) | `789eb491a748a9f2ecbbd02570e56ec325a1bcae714acdf347cf27112a457866` |
| `main_source.zip` | `2f754392983927d7e9e0436d19d0c86c8ad6c51f493b6ff2ce612cd336b5166c` |
| `supplement_source.zip` | `985c44c6656be870ec5c45fa87533b9e9f5fb1dc4bf52b6ec682a8d79433a1ba` |

All 47 rendered pages match clean source-archive builds at 85 DPI
with the separately supplied style. The changed main page and new
supplement section were inspected visually. The unchanged anonymous
replay material does not replace the checkpoint replay for this
experiment. No PDFs, archives, datasets or weights are committed.
This is a working draft, not submitted or independently confirmed.

[Whole-query experiment and replay](../moving_cv_slate_supervision_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
