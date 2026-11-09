# Evidence Manuscript v126

This version reports the fitting-only learned admission comparison and
the fixed-nominee bottleneck. The supplement distinguishes an empty
calibrated policy, a matched-count diagnostic, and a hindsight witness.
No A or independent confirmation outcomes were opened for this study.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v126
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v126.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v126.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` (8 pages) | `89384717c81589f51deff26f72b8c68ea35f93c6d7b425aaade463e5482c1a96` |
| `supplement/supplement.pdf` (36 pages) | `096800a94553762301a4040217054c4e8263f13a1cdf7f9f4becec749895efa3` |
| `main_source.zip` | `7eea4b79257b3b90fa4330a1589106161be97fb41fcd5ded97af6f27c557fa86` |
| `supplement_source.zip` | `b462dedd6d08dde5b1d51a09f39f5cf8d6edf57d463f80e2a0985244e80d68e4` |

[Experiment and replay](../moving_cv_admission_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
All 44 pages matched clean source-archive builds at 85 DPI with the same
separately supplied CVPR style. The new main and supplement paragraphs
were visually checked.
No PDF, raw trajectories, or model weights are committed. Not submitted.
