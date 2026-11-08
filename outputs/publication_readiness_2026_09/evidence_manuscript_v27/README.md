# OpenDD Development Diagnostic in the Supplement

This version adds Section 11 and Table 8 on page 6 of the anonymous
working supplement. It keeps the main draft and the original risk
requirements unchanged. The OpenDD study is explicitly separated from
independent confirmation and from the image-pixel stride-one task.
It includes the [primary OpenDD citation](https://arxiv.org/abs/2007.08463).

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v27
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v27.py \
  --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The six-page PDF was compiled directly and from a clean extraction with
the same separately provided CVPR style. All six rendered pages matched;
the new table and citation were inspected. The existing line-number
package emits an encoding warning; the build otherwise completed.

- Source ZIP SHA-256: `369bb089d2e599f2502dda1b6c86f3abf4027d878123aae5b83687e885a0f565`
- Built PDF SHA-256: `80a9563f4b62517fe2aac139d22fdd675c42e97aa94e1099416946c410973577`

The archive excludes data, model weights, forecasts and the style file.
[Experiment](../opendd_query_rank_v1/README.md) |
[Chinese local and CREATE instructions](../evidence_manuscript_v12/operation_zh.md).
No external outcome readout or submission readiness is claimed.
