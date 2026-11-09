# Evidence manuscript v148

The main paper and supplement record the direct spline forecast,
matched neighbor ablation, candidate capacity and exploratory
return-to-OLS4 experiment. The conservative full-roster and original
risk tests do not support a deployment or independent-transfer claim.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v148
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v148.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v148.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` (8 pages) | `3f840648f55894ae0edb2cbe6e7af0eed544784fbb19f46cf9e3d80deceea5f1` |
| `supplement/supplement.pdf` (44 pages) | `01fbc3b741e1a031647a1ff9a6496f23873433b66b4ec9db472f90290881242b` |
| `main_source.zip` | `f4da89d4461c7dac6ac7283d6180c970a53ef78858123e1dc90eb93f344b8894` |
| `supplement_source.zip` | `803c0f08c60518e095cc01843085edeff9345947da42604ac9785d19775885ba` |

All 52 pages match clean source-archive builds at 85 DPI with the
separately supplied CVPR style. The changed pages were visually
checked. These archives reproduce the document, not the upstream
training or independent experiments. PDFs, source archives and
checkpoints remain local.

[Experiment](../moving_causal_velocity_spline_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md)
