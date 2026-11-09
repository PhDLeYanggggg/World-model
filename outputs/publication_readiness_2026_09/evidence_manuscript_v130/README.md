# Evidence Manuscript v130

The main text condenses the recent fitting-source evidence to the
coverage limit, the uncapped hindsight witnesses, and the risk
prediction gap. The supplement separates matched predicted risk,
realized-information attribution, and the uncapped solver continuation.
No calibration setting qualifies. The independent groups are unopened.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v130
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v130.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v130.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` (8 pages) | `1602414e4e0c5e41a6e860d3f4dd6f75895741fc1a5a6ee79daec0c561bb3b41` |
| `supplement/supplement.pdf` (37 pages) | `b2066befa0d854da663a17e0e7e6c5977d592668da7b271f911741924dea810a` |
| `main_source.zip` | `c53ab58b8a193e8ce4f86151c26713ab5a5c0e298ed9cc4bac5416186e37dc7b` |
| `supplement_source.zip` | `1f38132e383c99c08273f0d6345a093e6206d476bb61e4acc8c12a79f5a5b73a` |

All 45 pages match clean source-archive builds at 85 DPI using the
same separately supplied style. The new supplement results were
visually inspected. The scalar replay separately reproduces the
coverage comparison with anonymous recording statistics. No PDFs,
archives, raw data or weights are committed. This is a research draft,
not a submitted paper or independent confirmation.

[Pooled coverage, attribution and replay](../moving_cv_pooled_coverage_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
