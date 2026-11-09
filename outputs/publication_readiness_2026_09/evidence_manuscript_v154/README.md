# Evidence manuscript v154

The paper distinguishes a failed transfer of the learned decision
head from a limit of the target action budget. The supplement
reports the matched OpenDD-to-Moving experiment and the utility-only
bound for arbitrary paths in one query. Neither supplies independent
confirmation or qualifies the method for deployment.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v154
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v154.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v154.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` (8 pages) | `f2b930efd61522d679d4b77ea229777d6d274788631cd9066c7ea512f0ae37da` |
| `supplement/supplement.pdf` (46 pages) | `5c30bfeff7d9d9d48eae11e31afa1f64bef727c3467e03dcb47f8caf8c868c3c` |
| `main_source.zip` | `41b6143cd234b6c84cd678bfd212bc06d49dc0354bd0d6ce58ea6bb09ce7e606` |
| `supplement_source.zip` | `46d1c003c30dd013143dba5d63cc5095ba67e53fa3adc79c583ebee1d5358fff` |

All 54 pages match clean source-archive builds at 85 DPI with the
separately supplied CVPR style. Changed pages were visually checked.
PDF and ZIP copies remain local. The official [dates](https://cvpr.thecvf.com/Conferences/2027/Dates)
still list registration on 10 November and full submission on 16
November 2026 (AoE) as checked on 9 November. Nothing was registered
or submitted in this experiment.

[Transfer experiment](../cross_domain_pass_transition_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md)
