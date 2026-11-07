# Decision Manuscript v20

The six-page anonymous working draft combines the query-local
directional result with the [recording feasibility bound](../native_recording_blend_feasibility_v1/README.md).
The new analysis rules out a particular family on exposed development
recordings. It does not supply independent confirmation, and the
manuscript is not ready to submit.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v20
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v20.py \
  --source-zip "$out/manuscript_source.zip"
cd "$out/build"
tectonic -X compile manuscript.tex
```

The builder binds the calculation to its exact replay receipt.
The anonymous archive excludes the official style, data, weights and
forecasts. With the verified style restored, a clean-directory build
matches all six page rasters. Source ZIP SHA-256:
`bcc8e6d24f7d0844a16de242d6b48eb68b9e23f60217cfaf6e66506d6292fa00`.

[Previous draft](../evidence_manuscript_v19/README.md) |
[Latest companion supplement](../evidence_manuscript_v21/README.md) |
[Chinese local/CREATE guide](../evidence_manuscript_v12/operation_zh.md).
