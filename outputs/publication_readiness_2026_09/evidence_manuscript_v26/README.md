# Observation Support And Conditional Gain

This six-page appendix adds the stepwise support and conditional
gain experiments to [v25](../evidence_manuscript_v25/README.md).
The main draft is unchanged. Table 7 on page 6 reports both the
improvement in label-support prediction and the failure to establish
better trajectory decisions at matched predicted support. The source
builder binds both scoring and replay receipts.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v26
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v26.py \
  --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The anonymous source archive rebuilt from a clean directory with the
same verified CVPR style. All six rendered pages matched and the new
table was inspected. The bundled line-number package still emits an
encoding warning. The source archive excludes images, private data,
forecasts, model weights and the separately distributed style file.

- Archive SHA-256: `cbff00b90e51d5a728684d3c8127e7d47c52f05d43578140d13909770edb967f`
- Built PDF SHA-256: `70a08dd387d82a56c606ebe467aed99f2cde275c8c2a1ab78d62b9b075662341`

[Chinese local and CREATE instructions](../evidence_manuscript_v12/operation_zh.md).
No new calibration or confirmation was opened. This is a working
appendix, not evidence that the paper is ready to submit.
