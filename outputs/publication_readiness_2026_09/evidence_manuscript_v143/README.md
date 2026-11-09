# Evidence manuscript v143

The supplement adds risk-boundary supervision with one-action
training substitutions, an accuracy-only control, and matched
held interventions. The main text is unchanged from
[v142](../evidence_manuscript_v142/README.md).

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v143
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v143.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `supplement/supplement.pdf` (42 pages) | `096543b7df4cf221962db5d21be93f4f9a1b6a7014e73f51296ba8908185fe85` |
| `supplement_source.zip` | `7147c70cb51c0a8d242c417cfc168953ddde7f0bd8d6964f90d9fab9063567f4` |
| `../moving_cv_risk_crossing_v1/anonymous_risk_crossing.zip` | `19cbb15a0ea311404193af24c5c42fcbf15236f909e4f8dedca89ef2ab7a31de` |

All 42 page renders match the clean source-archive build at
85 DPI with the separately supplied style. Pages 41 and 42 were
inspected. The anonymous archive reproduces the scalar risk and
accuracy table after clean extraction. It does not contain raw
trajectories or reproduce training. PDF, ZIP and checkpoints stay local.

[Experiment](../moving_cv_risk_crossing_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
