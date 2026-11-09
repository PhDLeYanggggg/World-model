# Evidence manuscript v146

The supplement adds fixed-count, fixed-predicted-risk gain learning
and a fractional capacity bound against OLS4. The main text is
unchanged from [v145](../evidence_manuscript_v145/README.md).

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v146
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v146.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
| --- | --- |
| `supplement/supplement.pdf` (43 pages) | `38799fa9c1c39e2c0e141a9738d3d8124e4d0e5ee4c286b6aeb0603895337388` |
| `supplement_source.zip` | `17102a25a2feb94f8c94aa00197fdd97e171b852e5c9c87da41d6222941cf8fb` |

All 43 pages match the clean source-archive render at 85 DPI
with the separately supplied style. The new table on page 42
was visually inspected. The PDF, archive and weights stay local.

[Experiment](../moving_cv_fixed_budget_gain_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
