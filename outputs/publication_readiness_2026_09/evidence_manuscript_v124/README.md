# Evidence Manuscript v124

The main paper now reports confirmed positive combinations that require the
expanded frozen candidate set at the same count cap. The supplement adds
exact witness checks, same-recipient and same-peak analytic rematching,
and the negative pair-geometry diagnostic. These are exposed development
results, not a causal decision gain or an independent result.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v124
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v124.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v124.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` (8 pages) | `8b96551b22fb12364f660ff413547e7a1b15ee9a4f574bdd4e6969ae01dccdd7` |
| `supplement/supplement.pdf` (35 pages) | `611a7ea47a5337c3ac95a614c6172026f7a5de188065f0b2210fc28f18d3373f` |
| `main_source.zip` | `3570a7dc4eaa242cb1d1c8b747928802ef7c3aaf7f4f04f57d44c2ce5197a3d8` |
| `supplement_source.zip` | `eb2b568470a1d107ca94853a31300255a2756cf6c618268aa44ab98de74ab6fe` |

[Experiment and witness replay](../moving_cv_joint_feasibility_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
All 43 pages matched clean source-archive builds at 85 DPI using the same
separately supplied CVPR style. The new main paragraph and supplement
results page were visually checked. No
PDF, raw data or private model is committed. Not submitted.
