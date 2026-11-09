# Evidence manuscript v137

The main text states the fixed-recipient exclusion result. The
supplement gives the equal-peak training comparison and the
arbitrary-path upper envelope, with the unchanged metric and risk
rule. No strategy qualified for independent confirmation.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v137
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v137.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v137.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` (8 pages) | `6e17ea216dd348cf0d95d52fdaae2762db6aed9f24394a6ae1ddabbfcc1f76be` |
| `supplement/supplement.pdf` (40 pages) | `2f6d237d7c2fd748690e46f95cd1a858b060b5f2cf2187822ea901ffcdd2b208` |
| `main_source.zip` | `0832a1f091b39c4ecae826f53a87faba7a819c73e7ef5d020d7aafa2d9127bf4` |
| `supplement_source.zip` | `6d2ae89061df6807002ef81b58b7c593b3510053ba9cdf5d1a93c6945f9aad56` |
| `../moving_cv_fixed_recipient_support_v1/anonymous_fixed_recipient_support.zip` | `75841ce78e454818926263032961831ec0002eb96d039a5e69228f91f62dca99` |

All 48 rendered pages match clean source-archive builds at 85 DPI
using the separately supplied style. The changed pages were
inspected. The NumPy archive reproduces the scalar policy table
after clean extraction, not model training or geometry. PDFs
and archives remain local. No submission action was taken.

[Fixed-recipient experiment](../moving_cv_fixed_recipient_support_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
