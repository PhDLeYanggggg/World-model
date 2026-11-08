# Forecast Transport and Realized Intervention Size

Section 14 and Table 11 report the frozen negative TUMDOT-ING transfer,
63 fresh OpenDD fits, and the equal-radius control. The main draft is
unchanged. The section distinguishes selective prediction with an explicit
reject option ([SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html))
from comparison against an emitted fallback, and does not transfer the
batch RL guarantees of [SPIBB](https://proceedings.mlr.press/v97/laroche19a.html)
to trajectory prediction.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v30
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v30.py \
  --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

Rebuilding the source ZIP with the same CVPR style produced eight identical
page renders. The added section, table, citations and LLM Usage Statement
were inspected. The existing style line-number encoding warning remains.

- Source ZIP SHA-256: `20317e979603adf77e811b32311e9cc1aac8d24f17e22f852d5ffacd6454bbcc`
- Built PDF SHA-256: `d1078e0b0887dd290714321bf722fd7b0cb4a186873880a8d661b6132a713cc4`

[OpenDD experiment](../opendd_hindcast_residual_v1/README.md) |
[TUMDOT-ING transfer](../tumdot_ing_frozen_transfer_v1/README.md) |
[Chinese local and CREATE instructions](../evidence_manuscript_v12/operation_zh.md).

The [official CVPR dates](https://cvpr.thecvf.com/Conferences/2027/Dates)
still list November 10 for registration and November 16 for full papers
(Anywhere on Earth). This supplement is verified draft material, not a
submission or evidence that the independent efficacy and safety gates pass.
