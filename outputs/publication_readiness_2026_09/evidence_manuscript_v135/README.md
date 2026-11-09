# Evidence manuscript v135

The main text states the 25% recording-budget result. The
supplement gives the cardinality certificate, budget table,
matched three-seed teacher ablation and the recipient/displacement
failure controls. These are development results. No model was
qualified for independent confirmation.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v135
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v135.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v135.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` (8 pages) | `70addcc3673998dd63bc0611b4dae185cde4cae77c4ff5e08b440c9d7d0b0bff` |
| `supplement/supplement.pdf` (40 pages) | `b1dfeb4c21375e4991908626853977c22964a533ccc638ab2a214bcc17fbe1d1` |
| `main_source.zip` | `b344be4c5baddd9f0df8313c1439838cdea8b10f5869f616ea7f7a7a94f3b852` |
| `supplement_source.zip` | `7501d1a7aed74d6b66ff8758acd439234074626817b29ae9ea99977519792bae` |
| `../moving_cv_minimum_budget_v1/anonymous_minimum_budget.zip` | `12d09b46ee3e66ea03cea9910249155892ecfeda2ec26d22dddd73dcd40394d1` |

All 48 rendered pages match clean source-archive builds at 85 DPI
with the separately supplied style. The changed pages and new
supplement table were inspected. A clean extraction verified the
anonymous scalar archive, not the training or path-level solver.
PDFs and archives remain local. No submission action was taken.

[Budget and teacher results](../moving_cv_minimum_budget_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
