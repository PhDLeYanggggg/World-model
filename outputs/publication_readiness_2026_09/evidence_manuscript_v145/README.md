# Evidence manuscript v145

The supplement adds direct versus conditional risk moments,
one-block swaps, and a low-coverage exact-query-count comparison
of joint versus independent choice. The main text is unchanged
from [v144](../evidence_manuscript_v144/README.md).

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v145
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v145.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `supplement/supplement.pdf` (42 pages) | `6ac07ae42692e659b92b27b3ed8b14032d6d69e3a8a1ef2cb93e5de615ecd4d8` |
| `supplement_source.zip` | `5fadeafbabce100af729656c555912c519c3a335c34b8932b7733539ba68ce91` |

All 42 pages match the clean source-archive render at 85 DPI
using the separately supplied style. Page 42 was inspected.
PDF, source archive and weights remain local.

[Experiment](../moving_cv_risk_components_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
