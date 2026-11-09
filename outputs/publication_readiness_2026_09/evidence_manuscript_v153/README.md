# Evidence manuscript v153

The main draft records that the whole-record reference target did
not improve over the corresponding local guard. The supplement
separates direct, factorized and matched-gain comparisons from the
single-query hindsight capacity. It is not a submission-ready result.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v153
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v153.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v153.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
| --- | --- |
| `build/manuscript.pdf` (8 pages) | `b42485b24dab9f826ce157cc653cd1ca8e86804263b42945de1bc36ab8f4ae32` |
| `supplement/supplement.pdf` (46 pages) | `57e63d80f17dcc8de623a453eb1ba943a07ead8e99c8b93281f460f2d2cd3652` |
| `main_source.zip` | `d20d484460d58735467c516bab7c06b3dec69aca0448943fc80b57a4ca388779` |
| `supplement_source.zip` | `b8f10a4d40f605f2ca7adbbe2c730a88c58a8552ed8071848bde859de1725547` |

All 54 pages match clean source-archive builds at 85 DPI with the
separately supplied CVPR style. The changed results and table were
visually checked. PDF and ZIP copies remain local.

[Direct experiment](../opendd_counterfactual_support_v1/README.md) |
[Factorized follow-up](../opendd_factored_counterfactual_support_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md)
